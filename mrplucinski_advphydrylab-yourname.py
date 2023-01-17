# Always start by importing the analysis packages we'll be using
# Notice that any line that stars with a " # " symbol is treated as a "comment" in Python, meaning it's for our reference and is not treated as code.

import pandas as pd  # Helps with organizing and formatting data.  The "pd" is a shorthand reference we can refer to this package as
import numpy as np   # "Num Py" is a package that contains common mathematical and statistical functions
import matplotlib as mpl # "Mat Plot Lib" is a package that helps with plotting and graphing data
import matplotlib.pyplot as plt  #  "Py Plot" is a sub-package of Mat Plot Lib that particularly helps with plotting and graphing data

# Also, don't forget to "run" this block of code to actually import the packages
# To "Run" a code block, make sure the cursor is in it and then press "shift" and "enter" at the same time
# You can also an option from the "Runtime" menu too.  When running this block, it loads the pacages in the background, so no output will be seen
data = pd.DataFrame(
    { "d": [1.5, 2.0, 3.0, 5.0, 1.5, 2.0, 3.0, 5.0, __________ ],
     "h": [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 10.0, 10.0, _______],
     "t": [73.0, 41.2, 18.4, 6.8, 43.5, 23.7, 10.5, 3.9, ______]
     })

# To check to see if the data was entered correctly, you can use the "head" function to print a few lines
data.head(5)
# Use "shift and enter" to run the code and see the output
xMin = 0
yMin = 0
h30 = data.loc[data["h"] == 30.0]
print(h30.head(5)) # view the data to show the 30cm height data has been isolated
plt.scatter(x = h30["d"], y = h30["t"])
plt.axis(xmin = xMin, ymin = yMin)
plt.title("Time vs Diameter, Height = 30 cm")
plt.xlabel("Diameter (cm)")
plt.ylabel("Time (s)")
plt.show()
data["inverseD"] = 1 / data["d"]
data.head(5)
# Plot the data again, focusing on the 30.0 cm constant height.  Copy and Paste are your friends here...
xMin = 0
yMin = 0
h30 = data.loc[data["h"] == 30.0]
print(h30.head(5)) # view the data to show the 30cm height data has been isolated
plt.scatter(x = h30["inverseD"], y = h30["t"])
plt.axis(xmin = xMin, ymin = yMin)
plt.title("Time vs Inverse Diameter, Height = 30 cm")
plt.xlabel("Inverse Diameter (1/cm)")
plt.ylabel("Time (s)")
plt.show()
data["inverseSquaredD"] = ___________  # Note:  use a "double star" operator to raise a value.  Squaring "x" would be: x**2
data.head(5)
xMin = 0
yMin = 0
h30 = data.loc[data["h"] == 30.0]
print(h30.head(5)) # view the data to show the 30cm height data has been isolated
plt.scatter(x = h30["inverseSquaredD"], y = h30["t"])
plt.axis(xmin = xMin, ymin = yMin)
plt.title("Time vs Inverse Diameter Squared, Height = 30 cm")
plt.xlabel("Inverse Diameter Squared (1/cm)^2")
plt.ylabel("Time (s)")
plt.show()
xMin = 0
yMin = 0
h30 = data.loc[data["h"] == 30.0]
h10 = data.loc[data["h"] == 10.0]
# include the other 2 height isolations

plt.scatter(x = h30["inverseSquaredD"], y = h30["t"], label = "h = 30 cm")
plt.scatter(x = h10["inverseSquaredD"], y = h10["t"], label = "h = 10 cm")
# plot the other 2 heights too

plt.axis(xmin = xMin, ymin = yMin)
plt.title("Time vs Inverse Diameter Squared")
plt.xlabel("Inverse Diameter Squared (1/cm)^2")
plt.ylabel("Time (s)")
plt.legend()
plt.show()  # Notice, putting multiple "scatter" calls will put them on the SAME plot, as long as they occur BEFORE a call to "show"
# Create a dataframe that called "d1_5" that isolates the data where the diameter is 1.5 cm

# Graph Time vs Height for a constant 1.5 cm diameter

# Create a new column in the data that manipulates the Height values to attempt to achieve a proportional relationship between Time and Height

# Graph Time vs the adjusted Height column, isolating only the 1.5 cm diameter data, checking for a proprotional result

# If a proprotional result has been reached, then graph Time vs adjusted Height for ALL of the data, but with the data separated into
# sets that each have constant diamters

# Combined Graph of your Data and Best Fit Line and Equation

# The below set of code graphs a linear, best fit line, of a sample data set.  It also produces the resulting linear equation as well.

sampleData = pd.DataFrame(
    { "sampleX": [1, 2, 3, 4, 5, 6, 7, 8],
     "sampleY": [4, 8, 12, 16, 20, 24, 28, 32]
     })

xMin = 0
yMin = 0
xMax = np.max(sampleData["sampleX"])

plt.scatter(x = sampleData["sampleX"], y = sampleData["sampleY"])
plt.axis(xmin = xMin, ymin = yMin)
plt.title("Sample Plot")
plt.xlabel("Sample X")
plt.ylabel("Sample Y")

# Code to create best fit line is here
slope, intercept = np.polyfit(sampleData["sampleX"], sampleData["sampleY"], 1)
xValues = np.arange(xMin, xMax, (xMax - xMin)/200) # Creates a set of 200, evenly spaced "x" values
plt.plot(xValues, slope*xValues + intercept, color = "r") # Plots the x values and calculates the "y" values based on the best fit line result
plt.show()

print("y=%.3fx+%.3f"%(slope, intercept)) #Code to print the resulting best fit equation.  Should show below the graph
