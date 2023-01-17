# Include pandas to manipulate dataframes and series

import pandas as pd

# Include matplotlib to plot diagrams

import matplotlib.pyplot as plt

%matplotlib inline



# Read human resources data from CSV data file

hr_data = pd.read_csv("../input/HR_comma_sep.csv")
# Print the shape of the dataset (number of samples and features)

print(hr_data.shape)

# Print the features of the dataset

features_name = hr_data.columns.tolist()

print(features_name)
# Reorganise the dataset to put the 'left' feature at the end

features_name.remove('left')

organized_features_name = features_name + ['left']

hr_data = hr_data[organized_features_name]

# Print the reorganised dataset first values

hr_data.head()
# Define sub-dataset with people who stayed and those who left

present_people_data = hr_data[hr_data['left'] == 0]

left_people_data = hr_data[hr_data['left'] == 1]
# Print the correlation matrix related to the 'left' feature of the dataset

hr_data.corr()['left']
# Define a figure with one diagram

fig = plt.figure(figsize=(6, 3))

present_people_plot = fig.add_subplot(121)

left_people_plot = fig.add_subplot(122)



# Draw an histogram of the present employees work accidents

present_people_plot.hist(present_people_data.Work_accident, 50, facecolor='blue')

present_people_plot.set_xlabel('Work accidents')

present_people_plot.set_title("Employees who stayed")

# Add a vertical line representing the mean work accidents for this sample

present_people_plot.axvline(present_people_data.Work_accident.mean(), color='r',

                            linestyle='dashed', linewidth=2)



# Draw an histogram of the left employees work accidents

left_people_plot.hist(left_people_data.Work_accident, 50, facecolor='red')

left_people_plot.set_xlabel('Work accidents')

left_people_plot.set_title("Employees who left")

# Add a vertical line representing the mean work accidents for this sample

left_people_plot.axvline(left_people_data.Work_accident.mean(), color='b',

                         linestyle='dashed', linewidth=2)



plt.show()
# Define a figure with one diagram

fig = plt.figure(figsize=(6, 3))

present_people_plot = fig.add_subplot(121)

left_people_plot = fig.add_subplot(122)



# Draw an histogram of the present employees time spent in company

present_people_plot.hist(present_people_data.time_spend_company, 50, facecolor='blue')

present_people_plot.set_xlabel('Time spent in company')

present_people_plot.set_title("Employees who stayed")

# Add a vertical line representing the mean time spent in company for this sample

present_people_plot.axvline(present_people_data.time_spend_company.mean(), color='r',

                            linestyle='dashed', linewidth=2)



# Draw an histogram of the left employees time spent in company

left_people_plot.hist(left_people_data.time_spend_company, 50, facecolor='red')

left_people_plot.set_xlabel('Time spent in company')

left_people_plot.set_title("Employees who left")

# Add a vertical line representing the mean time spent in company for this sample

left_people_plot.axvline(left_people_data.time_spend_company.mean(), color='b',

                         linestyle='dashed', linewidth=2)



plt.show()
from sklearn.ensemble import RandomForestClassifier



caracs_columns = hr_data[:][hr_data.columns[0:6]]

result = hr_data[:][hr_data.columns[9:]]



res = hr_data['left']

tt = hr_data.drop(['left', 'sales', 'salary'], axis=1)



random_forest_model = RandomForestClassifier(n_estimators=10)

random_forest_model.fit(tt, res)



importances = random_forest_model.feature_importances_

for f in range(tt.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, importances[f], importances[importances[f]]))