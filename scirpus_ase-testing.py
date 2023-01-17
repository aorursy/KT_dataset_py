import numpy as np

import pandas as pd

from ase import *
fn = "../input/train/{}/geometry.xyz".format(3)

atom_pos_data = {'Ga':[],'Al':[],'O':[]}

pos_data = []

lat_data = []

with open(fn) as f:

    for line in f.readlines():

        x = line.split()

        if x[0] == 'atom':

            atom_pos_data[x[4]].append([np.array(x[1:4], dtype=np.float)])

            pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])

        elif x[0] == 'lattice_vector':

            lat_data.append(np.array(x[1:4], dtype=np.float))
a = ase.Atoms()

for sym in ['Al','Ga','O']:

    positions = atom_pos_data[sym]

    for i in range(len(atom_pos_data[sym])):

        a.append(Atom(sym, (atom_pos_data[sym][i][0][0], atom_pos_data[sym][i][0][1], atom_pos_data[sym][i][0][2])))
a.set_cell([(lat_data[0][0], lat_data[0][1], lat_data[0][2]),

            (lat_data[1][0], lat_data[1][1], lat_data[1][2]),

            (lat_data[2][0], lat_data[2][1], lat_data[2][2])])
a.get_cell_lengths_and_angles()
a.get_volume()
a.get_chemical_symbols()
from IPython.display import HTML
def atoms_to_html(atoms):

    'Return the html representation the atoms object as string'



    from tempfile import NamedTemporaryFile



    with NamedTemporaryFile('r+', suffix='.html') as ntf:

        atoms.write(ntf.name, format='html')

        ntf.seek(0)

        html = ntf.read()

    return html
tbut_html = atoms_to_html(a)

HTML(tbut_html)