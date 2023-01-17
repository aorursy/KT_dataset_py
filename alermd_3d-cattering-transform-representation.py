import pandas as pd
import csv
import numpy as np
from scipy.spatial.distance import pdist

import seaborn as sns
import matplotlib.pyplot as plt
raw_data = pd.read_csv('../input/energy-molecule/roboBohr.csv')
raw_data.shape
raw_data.columns
raw_data.isnull().sum().sum()
Eat = raw_data['Eat']

Eat.describe()
sns.distplot(Eat, kde=True, color="g")
plt.xlabel('Atomization Energy')
plt.ylabel('Frequency')
plt.title('Atomization Energy Distribution');
ids = raw_data['pubchem_id'].values
nids = len(ids)
processed_data_path = '../input/processed-data/'

pos_vec = np.loadtxt(processed_data_path + 'pos_vec_pad.txt')
full_charges_vec = np.loadtxt(processed_data_path + 'full_charges_vec_pad.txt')

pos_vec = pos_vec.reshape(16242, 50, 3)
full_charges_vec = full_charges_vec.reshape(16242, 50)
# from pubchempy import get_compounds

# pos_vec = []
# full_charges_vec = []

# for i, cid in enumerate(ids):
#     print('Getting id {} of {}.'.format(i,  nids))

#     c = get_compounds(cid, 'cid', record_type='3d')[0]

#     pos = []
#     charges = []
#     for j, at in enumerate(c.atoms):
#         at = at.to_dict()
        
#         el = at['element']
#         number = at['number']
#         x, y, z = at['x'], at['y'], at['z']

#         pos.append([x, y, z])
#         charges.append(number)
    
#     pos_vec.append(pos)
#     full_charges_vec.append(charges)
# We zero-pad the position and full charge vectors

# maxlen = max([len(x) for x in pos_vec])

# pos_vec_pad = []
# for i in range(len(pos_vec)):
#     pos_vec[i] += [[0, 0, 0]] * (maxlen - len(pos_vec[i]))
#     pos_vec_pad.append(pos_vec[i])
    
# pos_vec_pad = np.array(pos_vec_pad)

# full_charges_vec_pad = []
# for i in range(len(full_charges_vec)):
#     full_charges_vec[i] += [0] * (maxlen - len(full_charges_vec[i]))
#     full_charges_vec_pad.append(full_charges_vec[i])
    
# full_charges_vec_pad = np.array(full_charges_vec_pad)
mask = full_charges_vec <= 2
valence_charges = full_charges_vec * mask

mask = np.logical_and(full_charges_vec > 2, full_charges_vec <= 10)
valence_charges += (full_charges_vec - 2) * mask

mask = np.logical_and(full_charges_vec > 10, full_charges_vec <= 18)
valence_charges += (full_charges_vec - 10) * mask
overlapping_precision = 1e-1
sigma = 2.0
min_dist = np.inf

for i in range(nids):
    n_atoms = np.sum(full_charges_vec[i] != 0)
    pos_i = pos_vec_pad[i, :n_atoms, :]
    min_dist = min(min_dist, pdist(pos_i).min())

delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
pos_vec_pad = pos_vec * delta / min_dist
M, N, O = 192, 128, 96

grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
grid = np.fft.ifftshift(grid)
from kymatio.torch import HarmonicScattering3D

# number of scales.
J = 2
# number of l values
L = 3
# List of exponents to the power of which moduli are raised before integration.
integral_powers = [0.5, 1.0, 2.0, 3.0]

scattering = HarmonicScattering3D(J=J,
                                  shape=(M, N, O),
                                  L=L,
                                  sigma_0=sigma, # bandwidth of mother wavelet
                                  integral_powers=integral_powers)
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

scattering.to(device)

print('Using', device)
batch_size = 8
n_batches = int(np.ceil(nids / batch_size))

print('n_batches = ', n_batches)
from kymatio.scattering3d.backend.torch_backend import compute_integrals
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians

order_0, orders_1_and_2 = [], []
print('Computing solid harmonic scattering coefficients of {} molecules from the QM7 database on {}'.format(
                                                            nids, {'cuda': 'GPU', 'cpu': 'CPU'}[device]))
print('sigma: {}, L: {}, J: {}, integral powers: {}'.format(sigma, L, J, integral_powers))

this_time = None
last_time = None
for i in range(n_batches):
    this_time = time.time()
    if last_time is not None:
        dt = this_time - last_time
        print("Iteration {} ETA: [{:02}:{:02}:{:02}]".format(
                    i + 1, int(((n_batches - i - 1) * dt) // 3600),
                    int((((n_batches - i - 1) * dt) // 60) % 60),
                    int(((n_batches - i - 1) * dt) % 60)))
    else:
        print("Iteration {} ETA: {}".format(i + 1, '-'))
    last_time = this_time
    time.sleep(1)

    # Extract the current batch.
    start = i * batch_size
    end = min(start + batch_size, nids)

    pos_batch = pos_vec_pad[start:end]
    full_batch = full_charges_vec_pad[start:end]
    val_batch = valence_charges[start:end]

    # Calculate the density map for the nuclear charges and transfer to PyTorch.
    full_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, full_batch, sigma)
    full_density_batch = torch.from_numpy(full_density_batch)
    full_density_batch = full_density_batch.to(device).float()

    # Compute zeroth-order, first-order, and second-order scattering coefficients of the nuclear charges.
    full_order_0 = compute_integrals(full_density_batch, integral_powers)
    full_scattering = scattering(full_density_batch)

    # Compute the map for valence charges.
    val_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, val_batch, sigma)
    val_density_batch = torch.from_numpy(val_density_batch)
    val_density_batch = val_density_batch.to(device).float()

    # Compute scattering coefficients for the valence charges.
    val_order_0 = compute_integrals(val_density_batch, integral_powers)
    val_scattering = scattering(val_density_batch)

    # Take the difference between nuclear and valence charges, then
    # compute the corresponding scattering coefficients.
    core_density_batch = full_density_batch - val_density_batch

    core_order_0 = compute_integrals(core_density_batch, integral_powers)
    core_scattering = scattering(core_density_batch)

    # Stack the nuclear, valence, and core coefficients into arrays and append them to the output.
    batch_order_0 = torch.stack((full_order_0, val_order_0, core_order_0), dim=-1)
    batch_orders_1_and_2 = torch.stack((full_scattering, val_scattering, core_scattering), dim=-1)

    order_0.append(batch_order_0)
    orders_1_and_2.append(batch_orders_1_and_2)