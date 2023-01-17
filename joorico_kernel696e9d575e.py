import pandas as pd

pd.set_option("display.max_rows", 5)



reviews = pd.read_csv("../input/winemagdata130k/winemag-data-130k-v2.csv", index_col=0)





from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = ____



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
countries = ____



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = ____



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
centered_price = ____



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
bargain_wine = ____



# Check your answer

q5.check()
#q5.hint()

#q5.solution()
descriptor_counts = ____



# Check your answer

q6.check()
#q6.hint()

#q6.solution()
star_ratings = ____



# Check your answer

q7.check()
#q7.hint()

#q7.solution()