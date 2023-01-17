import sqlite3
import pandas as pd
import numpy as np
# using the SQLite Table to read data.
con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')
#data
data = pd.read_sql_query("""SELECT * FROM Reviews""",con)
print("Number of data points", data.shape)
data.head(3)
#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""",con)

# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partitionScore(x):
    if x<3:
        return "negative"
    else:
        return "positive"

#changing reviews with score less than 3 to be positive and vice-versa
actualscore = filtered_data['Score']
positivenegative = actualscore.map(partitionScore)
filtered_data['Score'] = positivenegative

print("Number of data points in our data", filtered_data.shape)
filtered_data.head(3)
display = pd.read_sql_query("""
SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)
FROM Reviews
GROUP BY UserId
HAVING COUNT(*)>1
""", con)
print(display.shape)
display.head()
duplicatedata = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE UserId = '#oc-R115TNMSPFT9I7'
""", con)
duplicatedata.head()
display['COUNT(*)'].sum()
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display.head()

#reviews with same parameters other than ProductId belonged to the same product just having different flavour or quantity.
#Hence in order to reduce redundancy eliminated the rows having same parameters.
#So, first sort the data according to ProductId and then just keep the first similar product review and delelte the others.
#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape
#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
weird_data = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND HelpfulnessNumerator > HelpfulnessDenominator
""", con)
weird_data
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
final.shape
#two rows were removed
