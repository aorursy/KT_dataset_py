import os

import shutil



FIXED_LABELS = "fixed_labels.csv"

REMOVE_LABELS = "removed_files.csv"

ADDITIONAL_DIR = "./additional/"



os.mkdir( ADDITIONAL_DIR + "remove" )

for csvfile in [FIXED_LABELS, REMOVE_LABELS]:

    with open(csvfile) as f:

        lines = f.readlines()[1:]

        for line in lines:

            filename, old_label, new_label = (x.strip() for x in line.split(','))

            try:

                shutil.move( ADDITIONAL_DIR + old_label + '/' + filename,

                             ADDITIONAL_DIR + new_label + '/' + filename)

                print("moved '{}' to '{}'".format(filename, new_label))

            except:

                print("Cannot move '{}' to '{}'".format(filename, new_label))

                continue