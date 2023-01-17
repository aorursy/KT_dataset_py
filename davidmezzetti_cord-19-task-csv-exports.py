import os

import os.path

import pandas as pd

import shutil



from IPython.display import display, FileLinks



def formatter(dirname, fnames, included_suffixes):

    names = []

    for name in sorted(fnames):

        if name.endswith(".csv"):

            names.append("<a href='%s/%s'>%s</a><br/>" % (dirname, name, name))

    

    return names



def files(name):

    if not os.path.exists(name):

        # Copy files from input to output

        shutil.copytree(os.path.join("../input", name), name)



    # List directory contents

    display(FileLinks("%s" % name, recursive=False, notebook_display_formatter=formatter))
files("cord-19-population")
files("cord-19-relevant-factors")
files("cord-19-patient-descriptions")
files("cord-19-models-and-open-questions")
files("cord-19-materials")
files("cord-19-diagnostics")
files("cord-19-therapeutics")
files("cord-19-risk-factors")