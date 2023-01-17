import numpy as np

import pandas as pd

data = pd.read_csv("../input/udemy-courses/udemy_courses.csv")
delCols = ["course_title", "url"]

data.drop(labels=delCols, axis=1)



print( data.columns )

data.corr()
data = data[ ["course_id", "is_paid", "price", "num_subscribers", "num_reviews", "num_lectures", "content_duration"] ]

data.head()
data = data.groupby(by="course_id").mean()

data.head()
data.sort_values(by="num_reviews")[data["price"]==0]["price"].head()
data.sort_values(by="num_reviews").head()
print("List with id of the most engaging courses:")

print( list(data.sort_values(by="num_lectures").index[0:5]) )
print("Standard deviation for:")

print( "'price': "+str(data["price"].std()) )

print( "'num_subscribers': "+str(data["num_subscribers"].std()) )

print( "'num_reviews': "+str(data["num_reviews"].std()) )

print( "'num_lectures': "+str(data["num_lectures"].std()) )

print( "'content_duration': "+str(data["content_duration"].std()) )
print("Mean for:")

print( "'price': "+str(data["price"].mean()) )

print( "'num_lectures': "+str(data["num_lectures"].mean()) )
benefits = data["price"].div(data["content_duration"]).sort_values().head()