import magic
import os
import glob
def check_lamb_data(directory):
    """Function to check the file type, presence of lamb data and
    header w/ Virginia USDA info for VA livestock aution reports"""
    for file in glob.glob(f"{directory}*.txt"):
        print(f"Validation info for {file}:")

        # check for file types
        file_type = magic.from_file(file)
        print(f"File type: \t {file_type}")

        # check for missing data (Slaughter Lambs)
        if 'lamb' in open(file).read().lower():
            print(f"Lamb data: \t Present")
        else:
            print(f"Lamb data: \t Missing")

        # get info about when the data was uploaded
        with open(file,"r") as f:
            for line in f.readlines(): 
                if line.startswith("Richmond, VA"):
                    print(f"File info: \t {line}")
# check our validation script
check_lamb_data("/kaggle/input/")