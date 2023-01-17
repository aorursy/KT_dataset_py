import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# ASE is on Gitlab and on Readthedocs.
import ase.db, ase.io
from ase import Atoms

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Connect to the ASE db.
# c = ase.db.connect('../input/train.db')
# There are some issues with writing to the db on Kaggle.
# I am skipping that part and appending atoms objects to a list instead.
images = []

# Iterate over folders containing geometry files.
for folder in range(1, 3):
    # Define lists for output.
    positions = []
    cell = []
    symbols = []
    #Open and parse file.
    with open("../input/train/" + str(folder) + "/geometry.xyz") as f:
        for row, line in enumerate(f):
            fields = line.split(' ')
            # Each file contains a 3 line header.
            if row < 3:
                continue
            # Then the unit cell.
            elif row < 6:
                cell.append(fields[1:4])
            # Then rows of atomic positions and chemical symbols.
            else:
                positions.append(fields[1:4])
                symbols.append(fields[4].replace('\n', ''))
    # Make an atoms object from each file.
    atoms = Atoms(positions=np.array(positions, dtype=float),
                  symbols=symbols,
                  cell=np.array(cell, dtype=float))
    index = folder-1
    # The following code snippet would write all of the data to an ASE db file using an SQLite3 backend,
    # but currently it fails.
    #c.write(atoms,
    #        xyz_id=int(train['id'][index]),
    #        bandgap_energy_ev=float(train['bandgap_energy_ev'][index]),
    #        formation_energy_ev_natom=float(train['formation_energy_ev_natom'][index]),
    #        spacegroup=int(train['spacegroup'][index]),
    #        percent_atom_al=float(train['percent_atom_al'][index]),
    #        percent_atom_ga=float(train['percent_atom_ga'][index]),
    #        percent_atom_in=float(train['percent_atom_in'][index]))
    images.append(atoms)
# An atoms object contains Z numbers and symbols.
atoms = images[0]
print(atoms.numbers)
print(atoms.cell)

# A bunch of methods for extracting information.
print(atoms.get_cell_lengths_and_angles())
print(atoms.get_chemical_symbols())
all_distances = np.array(atoms.get_all_distances())
pdf = np.histogram(all_distances)