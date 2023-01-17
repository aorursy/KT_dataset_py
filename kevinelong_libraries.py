# BEGIN KAGGLE SPECIFIC MODIFICATION
from shutil import copyfile
copyfile(src = "../input/reuseme.py", dst = "../working/reuseme.py")
# END KAGGLE SPECIFIC MODIFICATION


# LEAST SAFE
# from reuseme import *
# print(do_a_thing(11, 44))

# SAFER
# from reuseme import do_a_thing, URLS, Food
# print(do_a_thing(11, 44))


# SAFEST
import reuseme

from random import randint

print(reuseme.do_a_thing(11, 44))

for u in reuseme.URLS:
    print(u)

donut = reuseme.Food("Donut!")
print(donut.name)

