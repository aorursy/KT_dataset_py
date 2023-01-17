# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex7 import *

print("Setup Complete")
# Check your answer (Run this code cell to receive credit!)

# If the amount of leather used is determined at the start of each month, there is no data leakage.

# However, if production happens and then leather used is reported at the end of each month, this is impossible to use in a prediction model and counts as a leakage source.

q_1.solution()
# Check your answer (Run this code cell to receive credit!)

# It's possible that they don't order leather at the same time each month.

q_2.solution()
# Check your answer (Run this code cell to receive credit!)

# It doesn't seem like there is a problem with the model, all data used for the predictions should be available beforehand, so it seems like this model is accurate.

q_3.solution()
# Check your answer (Run this code cell to receive credit!)

# Infection rates for surgeons would be available before predictions need to be made, as well as which surgeon performed on

# a specific patient. 

q_4.solution()
# Fill in the line below with one of 1, 2, 3 or 4.

potential_leakage_feature = 2



# Check your answer

q_5.check()
#q_5.hint()

#q_5.solution()