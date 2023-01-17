from Bio.PDB import PDBParser

import sys

import Bio.PDB

import Bio.PDB.StructureBuilder

from Bio.PDB.Residue import Residue



import pandas as pd
class SloppyStructureBuilder(Bio.PDB.StructureBuilder.StructureBuilder):

    """Cope with resSeq < 10,000 limitation by just incrementing internally.



    # Q: What's wrong here??

    #   Some atoms or residues will be missing in the data structure.

    #   WARNING: Residue (' ', 8954, ' ') redefined at line 74803.

    #   PDBConstructionException: Blank altlocs in duplicate residue SOL

    #   (' ', 8954, ' ') at line 74803.

    #

    # A: resSeq only goes to 9999 --> goes back to 0 (PDB format is not really

    #    good here)

    """



    # NOTE/TODO:

    # - H and W records are probably not handled yet (don't have examples

    #   to test)



    def __init__(self, verbose=False):

        Bio.PDB.StructureBuilder.StructureBuilder.__init__(self)

        self.max_resseq = -1

        self.verbose = verbose



    def init_residue(self, resname, field, resseq, icode):

        """Initiate a new Residue object.



        Arguments:

        o resname - string, e.g. "ASN"

        o field - hetero flag, "W" for waters, "H" for

            hetero residues, otherwise blanc.

        o resseq - int, sequence identifier

        o icode - string, insertion code



        """

        if field != " ":

            if field == "H":

                # The hetero field consists of

                # H_ + the residue name (e.g. H_FUC)

                field = "H_" + resname

        res_id = (field, resseq, icode)



        if resseq > self.max_resseq:

            self.max_resseq = resseq



        if field == " ":

            fudged_resseq = False

            while self.chain.has_id(res_id) or resseq == 0:

                # There already is a residue with the id (field, resseq, icode)

                # resseq == 0 catches already wrapped residue numbers which

                # do not trigger the has_id() test.

                #

                # Be sloppy and just increment...

                # (This code will not leave gaps in resids... I think)

                #

                # XXX: shouldn't we also do this for hetero atoms and water??

                self.max_resseq += 1

                resseq = self.max_resseq

                res_id = (field, resseq, icode)  # use max_resseq!

                fudged_resseq = True



            if fudged_resseq and self.verbose:

                sys.stderr.write(

                    "Residues are wrapping (Residue "

                    + "('%s', %i, '%s') at line %i)."

                    % (field, resseq, icode, self.line_counter)

                    + ".... assigning new resid %d.\n" % self.max_resseq

                )

        residue = Residue(res_id, resname, self.segid)

        self.chain.add(residue)

        self.residue = residue





class SloppyPDBIO(Bio.PDB.PDBIO):

    """PDBIO class that can deal with large pdb files as used in MD simulations



    - resSeq simply wrap and are printed modulo 10,000.

    - atom numbers wrap at 99,999 and are printed modulo 100,000



    """



    # The format string is derived from the PDB format as used in PDBIO.py

    # (has to be copied to the class because of the package layout it is not

    # externally accessible)

    _ATOM_FORMAT_STRING = (

        "%s%5i %-4s%c%3s %c%4i%c   " + "%8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n"

    )



    def _get_atom_line(

        self,

        atom,

        hetfield,

        segid,

        atom_number,

        resname,

        resseq,

        icode,

        chain_id,

        element="  ",

        charge="  ",

    ):

        """ Returns an ATOM string that is guaranteed to fit the ATOM format.



        - Resid (resseq) is wrapped (modulo 10,000) to fit into %4i (4I) format

        - Atom number (atom_number) is wrapped (modulo 100,000) to fit into

          %5i (5I) format



        """

        if hetfield != " ":

            record_type = "HETATM"

        else:

            record_type = "ATOM  "

        name = atom.get_fullname()

        altloc = atom.get_altloc()

        x, y, z = atom.get_coord()

        bfactor = atom.get_bfactor()

        occupancy = atom.get_occupancy()

        args = (

            record_type,

            atom_number % 100000,

            name,

            altloc,

            resname,

            chain_id,

            resseq % 10000,

            icode,

            x,

            y,

            z,

            occupancy,

            bfactor,

            segid,

            element,

            charge,

        )

        return self._ATOM_FORMAT_STRING % args



def get_structure(pdbfile, pdbid="system"):

    return sloppyparser.get_structure(pdbid, pdbfile)
train_df = pd.read_json(r'/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

train_df.head()
# Get the id of the first sequence



rna_id = train_df.id.values[0]

print(rna_id)
# get the pdb file and parse the structure



sloppyparser = Bio.PDB.PDBParser(

    PERMISSIVE=True, structure_builder=SloppyStructureBuilder()

)



structure = sloppyparser.get_structure("MD_system", f"/kaggle/input/openvaccine3dstructures/{rna_id}.pdb")
# print out C1 atom coordinates of each nucleotide



for x in structure.get_atoms():

    if str(x) == "<Atom C1'>":

        print(str(x), x.parent.resname.strip(), x.coord)
import os

import pickle

import numpy as np
data_dir = r'/kaggle/input/stanford-covid-vaccine'

train = pd.read_json(os.path.join(data_dir, 'train.json'), lines=True)

test = pd.read_json(os.path.join(data_dir, 'test.json'), lines=True)



use_cols = ['id','sequence', 'structure']

all_samples = pd.concat([train[use_cols], test[use_cols]], ignore_index=True, sort=False)
sloppyparser = Bio.PDB.PDBParser(

    PERMISSIVE=True, structure_builder=SloppyStructureBuilder()

)
pdb_folder = r"/kaggle/input/openvaccine3dstructures/"



def _get_seq(structure):

    coord = []

    seq = []

    for x in structure.get_atoms():

        if str(x) == "<Atom C1'>":

            coord.append(x.coord)

            seq.append(x.parent.resname.strip())

    seq = "".join(seq)

    return coord, seq



def get_c1_coord(rna_id, sequence):

    full_path = os.path.join(pdb_folder, f"{rna_id}.pdb")

    if not os.path.exists(full_path):

        return "file doesn't exist", np.nan

    structure = sloppyparser.get_structure("MD_system", full_path)

    coord, seq = _get_seq(structure)

    if seq != sequence:

        print(f"rna_id {rna_id} sequence doesn't match")

        return "sequence doesn't match", seq

    return coord, seq



all_samples[['coord', 'C1_implied_sequence']] = all_samples.apply(lambda x: get_c1_coord(x['id'], x['sequence']), axis=1, result_type="expand")
def point_distance(a, b):

    return np.sqrt(np.sum((a-b)**2))



def calc_dist(coord):

    N = len(coord)

    result = np.zeros((N, N))

    for i in range(N):

        for j in range(i+1, N):

            d = point_distance(coord[i], coord[j])

            result[i][j] = d

            result[j][i] = d

    return result
dist = {}

for i in range(len(all_samples)):

    assert all_samples.sequence.values[i] == all_samples.C1_implied_sequence.values[i]

    rna_id = all_samples.id.values[i]

    coord = all_samples.coord.values[i]

    dist[rna_id] = calc_dist(coord)

    if i % 100 == 0:

        print(i, end=" ")
pickle.dump(dist, open(r"dist", "wb"))