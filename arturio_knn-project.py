import pandas

import math

with open("../input/APF_Results3.csv", 'r') as csvfile:

    schools = pandas.read_csv(csvfile)



# The names of all the columns in the data.

print(schools.columns.values)



x_school = str(input('Enter '))



# Select the target school from our dataset

selected_school = schools[schools["school"] == x_school].iloc[0]



# Choose only the numeric columns (we'll use these to compute euclidean distance)

distance_columns = ['enr', 'ec', 'el', 'dis', 'rac']



# Find the distance from each school in the dataset to the target school.

school_distance = schools.apply(euclidean_distance, axis=1)



# Select only the numeric columns from the school dataset

schools_numeric = schools[distance_columns]



# Normalize all of the numeric columns

schools_normalized = (schools_numeric - schools_numeric.mean()) / schools_numeric.std()



from scipy.spatial import distance



# Fill in NA values in school_normalized

schools_normalized.fillna(0, inplace=True)



# Find the normalized vector for target school.

school_normalized = schools_normalized[schools["school"] == x_school]



# Find the distance between target school and others.

euclidean_distances = schools_normalized.apply(lambda row: distance.euclidean(row, school_normalized), axis=1)



# Create a new dataframe with distances.

distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})

distance_frame.sort_values("dist", inplace=True)

# Find the most similar school to target school (the lowest distance to our school is the school itself, the second smallest is the most similar school)

second_smallest = distance_frame.iloc[1]["idx"]

most_similar_school = schools.loc[int(second_smallest)]["school"]

most_similar_school