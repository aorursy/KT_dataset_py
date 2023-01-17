#Load the standard Python data science packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
college_filepath = "../input/ISLR-Auto/College.csv"

college = pd.read_csv(college_filepath)
college.head()
college.set_index("Unnamed: 0", inplace = True)

college.head()
#Rename the index to be more descriptive

college.rename_axis(index = "College", inplace = True)

college.head()
# Have Pandas treat the zeroth column in the CSV file as the index

# Then give the index a more descriptive name.

pd.read_csv(college_filepath, index_col = 0).rename_axis(index = "College").head()
college.isnull().any()
# Generate numerical summary of all of the numerical variables in the college data set

college.describe()
# Since the private column is a categorical variable, it's more readable to describe it separately using groupby and count

college["Private"].groupby(by = college["Private"]).count()
# Generating the scatterplot matrix using label-based indexing

sns.pairplot(college.loc[:, "Apps":"Books"])
# Generating the scatterplot matrix using integer-based indexing

sns.pairplot(college.iloc[:, 1:11])
ax = sns.catplot(x = "Private", y = "Outstate", kind = "box", order = ["Yes", "No"], data = college)

# Seaborn returns an axis object, so we can set the label for the y-axis to be more descriptive

ax.set(ylabel = "Out-of-state tuition (dollars)")

plt.show()
# Create a new column called Elite and set the default value as "No"

college["Elite"] = "No"

# Select all rows (i.e. schools) with over 50% of their students coming from the top 10% of their high school class

# Set the value of the Elite column for those schools to "Yes"

college.loc[college["Top10perc"] > 50, "Elite"] = "Yes"
# Take the Elite column of the college data set, group by its values, and count the occurrences of each value

college["Elite"].groupby(by = college["Elite"]).count()
ax = sns.catplot(x = "Elite", y = "Outstate", kind = "box", order = ["Yes", "No"], data = college)

ax.set(ylabel = "Out-of-state tuition (dollars)")

plt.show()
# Create grid of plots (fig)

# ax will be an array of four Axes objects

# Set the figure size so the plots aren't all squished together

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))



# Create histogram for number of applicants across all colleges

sns.distplot(college["Apps"], kde = False, ax = axes[0, 0])

axes[0, 0].set(xlabel = "", title = "All colleges")



# Create histogram for number of applicants at private colleges

sns.distplot(college.loc[college["Private"] == "Yes", "Apps"], kde = False, ax = axes[0, 1])

axes[0, 1].set(xlabel = "", title = "Private schools")



# Create histogram for number of applicants at elite colleges

sns.distplot(college.loc[college["Elite"] == "Yes", "Apps"], kde = False, ax = axes[1, 0])

axes[1, 0].set(xlabel = "", title = "Elite schools")



# Create histogram for number of applicants at public colleges

sns.distplot(college.loc[college["Private"] == "No", "Apps"], kde = False, ax = axes[1, 1])

axes[1, 1].set(xlabel = "", title = "Public schools")



fig.suptitle("Histograms of number of applicants by school type")
# Generate numerical summary of applicants by public vs private school

college["Apps"].groupby(by = college["Private"]).describe()
# Generate numerical summary of applicants by elite vs non-elite school

college["Apps"].groupby(by = college["Elite"]).describe()
# Create grid of plots (fig)

# ax will be an array of four Axes objects

# Set the figure size so the plots aren't all squished together

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))



# Create histogram for instructional expenditure per student across all colleges

sns.distplot(college["Expend"], kde = False, ax = axes[0, 0])

axes[0, 0].set(xlabel = "", title = "All colleges")



# Create histogram for instructional expenditure per student at private colleges

sns.distplot(college.loc[college["Private"] == "Yes", "Expend"], kde = False, ax = axes[0, 1])

axes[0, 1].set(xlabel = "", title = "Private schools")



# Create histogram for instructional expenditure per student at elite colleges

sns.distplot(college.loc[college["Elite"] == "Yes", "Expend"], kde = False, ax = axes[1, 0])

axes[1, 0].set(xlabel = "", title = "Elite schools")



# Create histogram for instructional expenditure per student at public colleges

sns.distplot(college.loc[college["Private"] == "No", "Expend"], kde = False, ax = axes[1, 1])

axes[1, 1].set(xlabel = "", title = "Public schools")



fig.suptitle("Histograms of instructional expenditure (USD) per student by school type")
# Generate numerical summary of instructional expenditure per student by public vs private schools

college["Expend"].groupby(by = college["Private"]).describe()
# Generate numerical summary of instructional expenditure per student by elite vs non-elite schools

college["Expend"].groupby(by = college["Elite"]).describe()
# Create grid of plots (fig)

# ax will be an array of four Axes objects

# Set the figure size so the plots aren't all squished together

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))



# Create histogram for student-faculty ratio across all colleges

sns.distplot(college["S.F.Ratio"], kde = False, ax = axes[0, 0])

axes[0, 0].set(xlabel = "", title = "All colleges")



# Create histogram for student-faculty ratio at private colleges

sns.distplot(college.loc[college["Private"] == "Yes", "S.F.Ratio"], kde = False, ax = axes[0, 1])

axes[0, 1].set(xlabel = "", title = "Private schools")



# Create histogram for student-faculty ratio at elite colleges

sns.distplot(college.loc[college["Elite"] == "Yes", "S.F.Ratio"], kde = False, ax = axes[1, 0])

axes[1, 0].set(xlabel = "", title = "Elite schools")



# Create histogram for student-faculty ratio at public colleges

sns.distplot(college.loc[college["Private"] == "No", "S.F.Ratio"], kde = False, ax = axes[1, 1])

axes[1, 1].set(xlabel = "", title = "Public schools")



fig.suptitle("Histograms of student-faculty ratio by school type")
# Generate numerical summary of student-faculty ratio by public vs private schools

college["S.F.Ratio"].groupby(by = college["Private"]).describe()
# Generate numerical summary of student-faculty ratio by elite vs non-elite schools

college["S.F.Ratio"].groupby(by = college["Elite"]).describe()
# Make a column for non-tuition costs (room and board, books, and personal)

college["NonTuitionCosts"] = college["Room.Board"] + college["Books"] + college["Personal"]
# Side-by-side boxplots for public vs private schools

ax = sns.catplot(x = "Private", y = "NonTuitionCosts", kind = "box", order = ["Yes", "No"],data = college)

ax.set(ylabel = "Total non-tuition costs per year (dollars)")

plt.show()
# Generate numerical summary of non-tuition costs by public vs private schools

college["NonTuitionCosts"].groupby(by = college["Private"]).describe()
# Side-by-side boxplots for elite vs non-elite schools

ax = sns.catplot(x = "Elite", y = "NonTuitionCosts", kind = "box", order = ["Yes", "No"], data = college)

ax.set(ylabel = "Total non-tuition costs per year (dollars)")

plt.show()
# Generate numerical summary of non-tuition costs by elite vs non-elite schools

college["NonTuitionCosts"].groupby(by = college["Elite"]).describe()
# Make a column for the acceptance rate of each school

college["AcceptPerc"] = college["Accept"] / college["Apps"] * 100
# Side-by-side boxplots for public vs private schools

ax = sns.catplot(x = "Private", y = "AcceptPerc", kind = "box", order = ["Yes", "No"], data = college)

ax.set(ylabel = "Percent of applicants accepted")

plt.show()
# Generate numerical summary of acceptance rates by public vs private schools

college["AcceptPerc"].groupby(by = college["Private"]).describe()
# Side-by-side boxplots for elite vs non-elite schools

ax = sns.catplot(x = "Elite", y = "AcceptPerc", kind = "box", order = ["Yes", "No"], data = college)

ax.set(ylabel = "Percent of applicants accepted")

plt.show()
# Generate numerical summary of acceptance rates by elite vs non-elite schools

college["AcceptPerc"].groupby(by = college["Elite"]).describe()
# Create grid of plots (fig)

# ax will be an array of four Axes objects

# Set the figure size so the plots aren't all squished together

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))



# Create histogram for percent of alumni who donate across all colleges

sns.distplot(college["perc.alumni"], kde = False, ax = axes[0, 0])

axes[0, 0].set(xlabel = "", title = "All colleges")



# Create histogram for percent of alumni who donate at private colleges

sns.distplot(college.loc[college["Private"] == "Yes", "perc.alumni"], kde = False, ax = axes[0, 1])

axes[0, 1].set(xlabel = "", title = "Private schools")



# Create histogram for percent of alumni who donate at elite colleges

sns.distplot(college.loc[college["Elite"] == "Yes", "perc.alumni"], kde = False, ax = axes[1, 0])

axes[1, 0].set(xlabel = "", title = "Elite schools")



# Create histogram for percent of alumni who donate at public colleges

sns.distplot(college.loc[college["Private"] == "No", "perc.alumni"], kde = False, ax = axes[1, 1])

axes[1, 1].set(xlabel = "", title = "Public schools")



fig.suptitle("Histograms of percent of alumni who donate by school type")
# Generate numerical summary of percent of alumni who donate by public vs private schools

college["perc.alumni"].groupby(by = college["Private"]).describe()
# Generate numerical summary of percent of alumni who donate by elite vs non-elite schools

college["perc.alumni"].groupby(by = college["Elite"]).describe()
# Side-by-side boxplots for public vs private schools

ax = sns.catplot(x = "Private", y = "Grad.Rate", kind = "box", order = ["Yes", "No"], data = college)

ax.set(ylabel = "Graduation rate")

plt.show()
# Generate numerical summary of graduation rate by public vs private schools

college["Grad.Rate"].groupby(by = college["Private"]).describe()
# Side-by-side boxplots for elite vs non-elite schools

ax = sns.catplot(x = "Elite", y = "Grad.Rate", kind = "box", order = ["Yes", "No"], data = college)

ax.set(ylabel = "Graduation rate")

plt.show()
# Generate numerical summary of graduation rate by elite vs non-elite schools

college["Grad.Rate"].groupby(by = college["Elite"]).describe()
# Create pair of scatter plots analyzing the relationship between number of faculty with PhDs and graduation rates

# Include least squares regression lines to help distinguish between different facets of the data

# Use columns to distinguish between elite and non-elite schools

# Use hue to distinguish between public and private schools

g = sns.lmplot(x = "PhD", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"],

               markers = ["o", "x"], data = college)

g.set(xlabel = "Number of faculty with PhDs", ylim = (0, 120))

plt.show()
# Create pair of scatter plots analyzing the relationship between number of faculty with terminal degrees and graduation rates

# Use columns to distinguish between elite and non-elite schools

# Include least squares regression lines to help distinguish between different facets of the data

# Use hue to distinguish between public and private schools

g = sns.lmplot(x = "Terminal", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"],

                markers = ["o", "x"], data = college)

g.set(xlabel = "Number of faculty with terminal degrees", ylim = (0, 120))

plt.show()
# Create pair of scatter plots analyzing the relationship between student-faculty ratio and graduation rates

# Include least squares regression lines to help distinguish between different facets of the data

# Use columns to distinguish between elite and non-elite schools

# Use hue to distinguish between public and private schools

g = sns.lmplot(x = "S.F.Ratio", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"],

                markers = ["o", "x"], data = college)

g.set(xlabel = "Student-faculty ratio", ylim = (0, 120))

plt.show()
# Create pair of scatter plots analyzing the relationship between instructional expenditure per student and graduation rates

# Include least squares regression lines to help distinguish between different facets of the data

# Use columns to distinguish between elite and non-elite schools

# Use hue to distinguish between public and private schools

g = sns.lmplot(x = "Expend", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"], 

               markers = ["o", "x"], data = college)

g.set(xlabel = "Instructional expenditure per student (USD)", ylim = (0, 120))

plt.show()
# Create variable for the name of the file containing the Auto data set

auto_filename = "../input/ISLR-Auto/Auto.csv"

# Load the Auto data set into a Pandas dataframe, treating question marks as na values

auto = pd.read_csv(auto_filename, na_values = ["?"])

# Drop the rows which contain missing values (safe to do since we've worked with this data in a previous lab)

auto.dropna(inplace = True)

# Check the dimensions of the dataframe

auto.shape
auto.head()
# Range = max - min

# Use the max() and min() functions on just the numeric data

# The argument axis = 0 means that we compute the max/min along each index

auto_max = auto.loc[:, "mpg":"year"].max(axis = 0)

auto_min = auto.loc[:, "mpg":"year"].min(axis = 0)

auto_range = auto_max - auto_min

# Generate a dataframe with the max, min, and range for each quantitative variable

pd.DataFrame({"max":auto_max, "min":auto_min, "range":auto_range})
# Compute mean of each quantitative variable

auto_mean = auto.loc[:, "mpg":"year"].mean(axis = 0)

# Compute standard deviation of each quantitative variable

auto_sd = auto.loc[:, "mpg":"year"].std(axis = 0)

# Generate a dataframe with the mean and standard deviation of each quantitative predictor

# Note that I also could have used the describe() function as well

pd.DataFrame({"mean":auto_mean, "std dev":auto_sd})
# Reset the index of the auto data frame

auto.reset_index(drop = True, inplace = True)

# Create dataframe in which the 10th through 85th observations are dropped

# Don't forget that Pandas dataframes are zero-indexed

auto_dropped = auto.drop(index = list(range(9, 85)))

# Compute max, min, range, mean, and standard deviation for each quantitative variable

dropped_max = auto_dropped.loc[:, "mpg":"year"].max(axis = 0)

dropped_min = auto_dropped.loc[:, "mpg":"year"].min(axis = 0)

dropped_range = dropped_max - dropped_min

dropped_mean = auto_dropped.loc[:, "mpg":"year"].mean(axis = 0)

dropped_sd = auto_dropped.loc[:, "mpg":"year"].std(axis = 0)

# Generate a dataframe with the max, min, range, mean, and standard deviation for each quantitative variable

# Again note that the describe() function would provide all of these values except for the range

pd.DataFrame({"max":dropped_max, "min":dropped_min, "range":dropped_range, "mean":dropped_mean, "std dev":dropped_sd})
# Convert the origin column from numerical codes to the meanings of each code

# 1 = American, 2 = European, 3 = Japanese

origin_dict = {1: "American", 2: "European", 3: "Japanese"}

auto["origin"] = auto["origin"].transform(lambda x: origin_dict[x]).astype("category")
# Create scatter plot for the relationship between engine displacement and mpg

# Use hue to highlight the origin of each car

g = sns.relplot(x = "displacement", y = "mpg", hue = "origin", data = auto)

g.set(xlabel = "Engine displacement (cubic inches)")

plt.show()
# Create scatter plot for the relationship between horsepower and mpg

# Use hue to highlight the origin of each car

g = sns.relplot(x = "horsepower", y = "mpg", hue = "origin", data = auto)

plt.show()
# Create scatter plot for the relationship between car weight and mpg

# Use hue to highlight the origin of each car

g = sns.relplot(x = "weight", y = "mpg", hue = "origin", data = auto)

g.set(xlabel = "Car weight (pounds)")

plt.show()
# Create scatter plot for the relationship between model year and mpg

# Use hue to highlight the origin of each car

g = sns.relplot(x = "year", y = "mpg", hue = "origin", data = auto)

g.set(xlabel = "Model year")

plt.show()
# Alternatively use pairplot to create scatterplots relating mpg to engine displacement, horsepower,

# car weight, and car manufacture year

# Use hue to highlight the origin of each car

g = sns.pairplot(auto, hue = "origin", y_vars = ["mpg"], x_vars = ["displacement", "horsepower", "weight", "year"],

                height = 5)
# Create scatter plot for the relationship between model year and acceleration

# Use hue to highlight the origin of each car

g = sns.relplot(x = "year", y = "acceleration", hue = "origin", data = auto)

g.set(xlabel = "Model year", ylabel = "0 to 60mph time (seconds)")

plt.show()
# Create scatter plot for the relationship between model year and engine displacement

# Use hue to highlight the origin of each car

g = sns.relplot(x = "year", y = "displacement", hue = "origin", data = auto)

g.set(xlabel = "Model year", ylabel = "Engine displacement (cubic inches)")

plt.show()
# Create scatter plot for the relationship between model year and car weight

# Use hue to highlight the origin of each car

g = sns.relplot(x = "year", y = "weight", hue = "origin", data = auto)

g.set(xlabel = "Model year", ylabel = "Car weight (pounds)")

plt.show()
# Create scatter plot for the relationship between model year and horsepower

# Use hue to highlight the origin of each car

g = sns.relplot(x = "year", y = "horsepower", hue = "origin", data = auto)

g.set(xlabel = "Model year", ylabel = "horsepower")

plt.show()
# Create scatter plot for the relationship between car weight and acceleration

# Use hue to highlight the origin of each car

g = sns.relplot(x = "weight", y = "acceleration", hue = "origin", data = auto)

g.set(xlabel = "Car weight (pounds)", ylabel = "0 to 60mph time (seconds)")

plt.show()
# Create scatter plot for the relationship between engine displacement and acceleration

# Use hue to highlight the origin of each car

g = sns.relplot(x = "displacement", y = "acceleration", hue = "origin", data = auto)

g.set(xlabel = "Engine displacement (cubic inches)", ylabel = "0 to 60mph time (seconds)")

plt.show()
# Create scatter plot for the relationship between horsepower and acceleration

# Use hue to highlight the origin of each car

g = sns.relplot(x = "horsepower", y = "acceleration", hue = "origin", data = auto)

g.set(xlabel = "Horsepower", ylabel = "0 to 60mph time (seconds)")

plt.show()
# Create swarm plot for the relationship between number of engine cylinders and acceleration

# Use hue to highlight the origin of each car

g = sns.catplot(x = "cylinders", y = "acceleration", hue = "origin", data = auto, kind = "swarm")

g.set(xlabel = "Number of engine cylinders", ylabel = "0 to 60mph time (seconds)")

plt.show()
# Alternatively use pairplot to create scatterplots relating acceleration to engine displacement, horsepower,

# car weight

# Use hue to highlight the origin of each car

g = sns.pairplot(auto, hue = "origin", y_vars = ["acceleration"], x_vars = ["displacement", "horsepower", "weight"],

                height = 5)
# Create scatter plot for the relationship between car weight and horsepower

# Use hue to highlight the origin of each car

g = sns.relplot(x = "weight", y = "horsepower", hue = "origin", data = auto)

g.set(xlabel = "Car weight (pounds)", ylabel = "Horsepower")

plt.show()
# Create scatter plot for the relationship between car weight and engine displacement

# Use hue to highlight the origin of each car

g = sns.relplot(x = "weight", y = "displacement", hue = "origin", data = auto)

g.set(xlabel = "Car weight (pounds)", ylabel = "Engine displacement (cubic inches)")

plt.show()
# Create box plot comparing the fuel effiency of American, European, and Japanese cars

g = sns.catplot(x = "origin", y = "mpg", data = auto, kind = "box")
auto.loc[:, "mpg":"acceleration"].groupby(auto["origin"]).agg(["mean", "std", "min", "median", "max"]).T
# Create variable for corrected Boston dataset file name

boston_filename = "../input/corrected-boston-housing/boston_corrected.csv"

# Load the data into a Pandas dataframe

# Create a multi-index on the TOWN and TRACT columns

boston = pd.read_csv(boston_filename, index_col= ["TOWN", "TRACT"])

boston.head()
boston.shape
# Since I won't be using them, I'll drop the TOWNNO, LON, LAT, and MEDV columns

boston.drop(columns = ["TOWNNO", "LON", "LAT", "MEDV"], inplace = True)

boston.head()
# Use pairplot to create a trio of scatterplots relating median home value with

# percent of home built prior to 1940, percent of lower socioeconomic status residents, and pupil-teacher ratio

g = sns.pairplot(boston, x_vars = ["AGE", "LSTAT", "PTRATIO"], y_vars = ["CMEDV"], height = 4)
# Use catplot to create a boxplot comparing the median home values between tracts

# bordering the Charles River and those which do not

g = sns.catplot(x = "CHAS", y = "CMEDV", kind = "box", order = [0, 1], data = boston)
# Use pairplot to create a pair of scatter plots to relate the concentration of nitric oxides with median home value

# and percent of non-retail business acres

g = sns.pairplot(boston, x_vars = ["CMEDV", "INDUS"], y_vars = ["NOX"], height = 4)
# Use pairplot to create a pair of scatter plots to relate the median home value to the proportion of black residents

# and the proximity to Boston employment centers

g = sns.pairplot(boston, x_vars = ["B", "DIS"], y_vars = ["CMEDV"], height = 4)
# Use pairplot to create a quartet of scatter plots to relate the per capita crime rate with proportion of Black residents,

# proportion of lower-status residents, median home value, and proximity to Boston employment centers

g = sns.pairplot(boston, x_vars = ["B", "LSTAT", "CMEDV", "DIS"], y_vars = ["CRIM"], height = 4)
# Create grid of plots (fig)

# ax will be an array of three Axes objects

# Set the figure size so the plots aren't all squished together

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 5))



# Create histogram for Boston crime rates

sns.distplot(boston["CRIM"], kde = False, ax = axes[0])

axes[0].set(xlabel = "", title = "Histogram of per capita crime rate")



# Create histogram for Boston tax rates

# Use more bins than the default given by the Freedman-Diaconis rule to see more of the shape of the distribution

sns.distplot(boston["TAX"], bins = 20, kde = False, ax = axes[1])

axes[1].set(xlabel = "", title = "Histogram of tax rate")



# Create histogram for Boston pupil-teacher ratios

sns.distplot(boston["PTRATIO"], kde = False, ax = axes[2])

axes[2].set(xlabel = "", title = "Histogram of pupil-teacher ratio")
boston.loc[:, ["CRIM", "TAX", "PTRATIO"]].describe()
# Use the fact that in the data set, 1 = Borders the Charles and 0 = otherwise to count the number

# of tracts which borders the Charles by summing along that column

boston["CHAS"].sum()
# We can use a boolean mask to only take the rows for tracts which border the Charles

# Then look at the 0-level of the multi-index to access the town names and return the unique ones

# We could then check the Index.size attribute to get the number of unique towns which border the Charles

boston[boston["CHAS"] == 1].index.unique(level = 0)
# Alternatively, we can get the level values for the 0-level of the multi-index and use the

# Index.nunique() function to return the number of unique towns which border the Charles

boston[boston["CHAS"] == 1].index.get_level_values(0).nunique()
boston["PTRATIO"].describe()
min_medv = boston["CMEDV"].min()

boston[boston["CMEDV"] == min_medv]
# Round to four decimal places so I don't have to scroll horizontally to view entire

# set of summary statistics

boston.describe().round(4)
# Use the fact that when summing, False has an integer value of 0 and True has an integer value of 1

(boston["RM"] > 7).sum()
# Use the fact that when summing, False has an integer value of 0 and True has an integer value of 1

(boston["RM"] > 8).sum()
boston.loc[boston["RM"] > 8]
boston.loc[boston["RM"] > 8].describe().round(4)