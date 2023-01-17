# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql_advanced.ex4 import *

print("Setup Complete")
# Fill in your answer

query_to_optimize = 3



# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
# Line below will give you a hint

#q_2.hint()
# View the solution (Run this code cell to receive credit!)

# It looks like the query pulls all of the data available and then joins it by owner ID. 

# Then it performs a function over all timestamp data from all owners and then joins it again.

# Then it finally checks for a specific owner of the pet in question.



# It should be much faster for the query to not join data twice and check all timestamps for all pets.

# The query should be optimized to only check for timestamp and location data for pets owned by MitzieOwnerID,

# rather than check timestamp and location data for all pets in the database and THEN narrow it down.

q_2.solution()