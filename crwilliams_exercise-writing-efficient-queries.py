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

q_1.hint()

q_1.solution()
"""

WITH LocationsAndOwners AS 

(

SELECT co.CostumeID, cl.Location 

FROM CostumeOwners co 

INNER JOIN CostumeLocations cl

   ON co.CostumeID = cl.CostumeID

WHERE OwnerID = MitzieOwnerID

),

LastSeen AS

(

SELECT CostumeID, MAX(Timestamp)

FROM LocationsAndOwners

GROUP BY CostumeID

WHERE OwnerID = MitzieOwnerID

)

SELECT lo.CostumeID, Location 

FROM LocationsAndOwners lo 

INNER JOIN LastSeen ls 

    ON lo.Timestamp = ls.Timestamp 

        AND lo.CostumeID = ls.CostumeID

"""
# Lines below will give you a hint or the solution

q_2.hint()

q_2.solution()