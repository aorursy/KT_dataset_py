# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/my_functions.py", dst = "../working/my_functions.py")

# import all our functions
from my_functions import *
# we can now use this function!
times_two_plus_three(4)
# and this one too!
print_cat()