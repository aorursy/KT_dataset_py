!pip install pywigxjpf argparse
#!git clone https://gitlab.com/AmosEgel/smuthi.git
!git clone https://gitlab.com/kostyfisik/smuthi.git
# !cd smuthi && git checkout ipython-terminal
!mkdir smuthi_nfmds_bin
!mkdir smuthi/smuthi_nfmds_bin
!cd smuthi && python setup.py install
!cd smuthi %% git branch
import sys
# import smuthi
default_stdout = sys.stdout
import matplotlib.pyplot as plt
def plt_write(text):
    plt.figure(figsize=(0.1,0.1))
#     plt.title(text)
    plt.annotate(text,(0,0))
    plt.axis('off')
    plt.margins(0.)
    plt.tight_layout()
    plt.show()
    
plt_write("test")
plt_write("test")
# !cd smuthi/examples/tutorials/sphere_on_substrate && python dielectric_sphere_on_substrate.py
import sys
# import smuthi
default_stdout = sys.stdout
sys.stdout = default_stdout
sys.stdout.write("test")
default_stdout.write("test2")

sys.stdout.write("test")
sys.stdout = default_stdout
sys.stdout.write("test")

sys.stdout = default_stdout
#*****************************************************************************#
# This is a simple example script for Smuthi v0.8.6.                          #
# It evaluates the differential scattering cross section of a large number of #
# glass spheres on a glass substrate, excited by a plane wave under normal    #
# incidence.                                                                  #
#*****************************************************************************#

import time
import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.scattered_field
import smuthi.graphical_output
import smuthi.coordinates as coord
import smuthi.cuda_sources as cu
import matplotlib.pyplot as plt

import sys
default_stdout = sys.stdout
# In this file, all lengths are given in nanometers

wl0 = 550
k0 = 2 * np.pi / wl0
period = wl0
radius = period / 6.
refractive_index = 3.464101615 + 0.j

N = 2

coord.set_default_k_parallel(vacuum_wavelength=wl0,
                             neff_resolution=5e-3,
                             neff_max=2.5,
                             neff_imag=1e-2)
def get_spheres_list(N, radius, period):
    # Scattering particles (NxN square matrix)
    spheres_list = []
    number_of_spheres = N**2
    xn = np.array(range(N))*period
    yn = xn
    z = np.zeros(N**2)
    x, y = np.meshgrid(xn,yn)
    x = x.flatten()
    y = y.flatten()
    for i in range(number_of_spheres):
        spheres_list.append(smuthi.particles.Sphere(position=[x[i], y[i], 2*radius],
                                                    refractive_index=refractive_index,
                                                    radius=radius,
                                                    l_max=3))
    return spheres_list


def simulate_N_spheres(spheres_list = [],
                       wl = 550,
                       solver_tolerance=5e-4,
                       lookup_resolution=5,
                       interpolation_order='linear'):
    number_of_spheres = len(spheres_list)
    default_stdout.write("spheres N: %i\n"%(number_of_spheres))
    # Initialize the layer system: substrate (glass) and ambient (air)
    two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                           refractive_indices=[1, 1])

    # Initial field
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
                                                polar_angle=np.pi,  # from top
                                                azimuthal_angle=0,
                                                polarization=0)  # 0=TE 1=TM



    preparation_time = 0
    solution_time = 0

    if number_of_spheres < lookup_resolution:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=spheres_list,
                                                  initial_field=plane_wave,
                                                  solver_type='gmres',
                                                  solver_tolerance=solver_tolerance,
                                                  store_coupling_matrix=True,
                                                  coupling_matrix_lookup_resolution=None,
                                                  coupling_matrix_interpolator_kind=interpolation_order,
                                                  log_to_file=False,
                                                  log_to_terminal=False)
        start = time.time()
        simulation.run()
        # sys.stdout = sys.__stdout__ # Restore output after muting it in simulation
        end = time.time()
        solution_time = end - start

    else:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=spheres_list,
                                                  initial_field=plane_wave,
                                                  solver_type='gmres',
                                                  solver_tolerance=solver_tolerance,
                                                  store_coupling_matrix=False,
                                                  coupling_matrix_lookup_resolution=lookup_resolution,
                                                  coupling_matrix_interpolator_kind=interpolation_order,
                                                  log_to_file=False,
                                                  log_to_terminal=False)
        
        start = time.time()
        simulation.initialize_linear_system()
        simulation.linear_system.prepare()
        end = time.time()

        preparation_time = end - start

        start = time.time()
        simulation.linear_system.solve()
        end = time.time()
        solution_time = end - start
        # sys.stdout = sys.__stdout__ # Restore output after muting it in simulation


    # compute cross section
    #     ecs = smuthi.scattered_field.scattering_cross_section(initial_field=plane_wave,
    #                                                           particle_list=spheres_list,
    #                                                           layer_system=two_layers)
    scattering_cross_section = smuthi.scattered_field.scattering_cross_section(initial_field=plane_wave,
                                                                               particle_list=spheres_list,
                                                                               layer_system=two_layers)
    Q_sca = (scattering_cross_section.top().integral()[0]
             + scattering_cross_section.top().integral()[1]
             + scattering_cross_section.bottom().integral()[0]
             + scattering_cross_section.bottom().integral()[1]).real/(np.pi*radius**2*number_of_spheres)

    plt_write("wl: %g, spheres N: %i, prep: %g s, solu: %g s\n"%( wl,
                                                                  number_of_spheres, preparation_time, solution_time))
    return [Q_sca, preparation_time, solution_time]


# launch a series of simulations:


# Initialize and run simulation
cu.enable_gpu(True)
if not cu.use_gpu:
    default_stdout.write("Failed to load pycuda")


# iterative solution on GPU
gpu_iterative_particle_N = [50]

gpu_iterative_times = []
gpu_iterative_preptimes = []
gpu_iterative_ecs = []
Q_spectra = []
WLs = np.linspace(wl0/1.4, wl0*1.5, 5)
start = time.time()
for N in gpu_iterative_particle_N:
    default_stdout.write("\n----------------------------------------------------------")
    default_stdout.write("Simulating %ix%i particles on GPU with iterative solver."%(N,N))
    spheres_list = get_spheres_list(N, radius, period)
    spectrum = []
    for wl in WLs:
        results = simulate_N_spheres(spheres_list = spheres_list, wl=wl)
        spectrum.append(results[0])
    Q_spectra.append(spectrum)
    plt.figure()
    plt.plot(WLs,spectrum)
    plt.show()
    gpu_iterative_times.append(results[1]+results[2])
    gpu_iterative_preptimes.append(results[1])
    gpu_iterative_ecs.append(results[0])
end = time.time()
total_time = end - start
plt_write(str(total_time))

# get GPU device name
import pycuda.driver as drv
drv.init()
device_name = drv.Device(0).name()

plt.figure()
plt.xlabel("Number of spheres")
plt.ylabel("Solver time")
plt.loglog(gpu_iterative_particle_N, gpu_iterative_times, '-gd')
plt.loglog(gpu_iterative_particle_N, gpu_iterative_preptimes, '--gd')
plt.legend(["iter., GPU, total", "iter., GPU, prep."])
plt.grid()
plt.savefig("runtime.png")

plt.figure()
plt.xlabel("Number of spheres")
plt.ylabel("Extinction cross section [nm^2]")
plt.loglog(gpu_iterative_particle_N, gpu_iterative_ecs, '-gd')
plt.legend(["Iterative solution on " + device_name])
plt.grid()
plt.savefig("cross_section.png")

plt.show()

from matplotlib import pyplot as plt

plt.figure()
plt.xlabel("Number of spheres")
plt.ylabel("Extinction cross section [nm^2]")
plt.loglog(gpu_iterative_particle_N, gpu_iterative_ecs, '-gd')
plt.legend(["Iterative solution on " + device_name])
plt.grid()
plt.savefig("cross_section.png")
!ls