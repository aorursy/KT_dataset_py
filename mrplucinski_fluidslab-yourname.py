# Import commonly used packages

import pandas as pd  # Helps with organizing and formatting data.  The "pd" is a shorthand reference we can refer to this package as
import numpy as np   # "Num Py" is a package that contains common mathematical and statistical functions
import matplotlib as mpl # "Mat Plot Lib" is a package that helps with plotting and graphing data
import matplotlib.pyplot as plt  #  "Py Plot" is a sub-package of Mat Plot Lib that particularly helps with plotting and graphing data

# Run the code block to make sure the packages are imported
print("Packages Imported!")
# Entering data:  Below is a sample data frame to remind you how to enter your raw data into Python
sampleData = pd.DataFrame(
    { "sampleX": [1, 2, 3, 4, 5, 6, 7, 8],
     "sampleY": [4, 8, 12, 16, 20, 24, 28, 32]
     })
sampleData.head(3)
# New Column in a data frame
sampleData["xSquared"] = sampleData["sampleX"]**2
sampleData.head(3)
# Graphing Data
xMin = 0
yMin = 0

plt.scatter(x = sampleData["sampleX"], y = sampleData["sampleY"])
plt.axis(xmin = xMin, ymin = yMin)
plt.title("Sample Plot")
plt.xlabel("Sample X")
plt.ylabel("Sample Y")
plt.show()
# Graphing with a Best Fit Line
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
# Create a dataFrame called:  dataHoleArea
# Title the columns similarly to the suggested data table in your analysis guide

# Add several columns to the dataHoleArea DataFrame (as described in your analysis guide).  Calculated colums to add are...
#   - Area of the holes, complete with unit conversion so the area is in centimaters squared
#   - Velocity of the fluid as it exits the hole.  Remember our projectile motion talk from the pre-lab and use 980 cm/s/s for the accel of gravity
#   - Fluid flow rate in cubic centimeters per second


# Graph and linearize the Velocity of the Fluid vs the Area of the Holes.  When completed, answer number 5 & 6 in your analysis guide

# Graph and linearize the Fluid Flow Rate vs the Area of the Hole.  When completed, answer number 7 in your analysis guide

# Create a dataFrame called:  dataHoleHeight
# Title the columns similarly to the suggested data table in your analysis guide.  Be sure any distance values are properly converted to centimeters

# Add several columns to the dataHoleHeight DataFrame (as described in your analysis guide).  Calculated colums to add are...
#   - Velocity of the fluid as it exits the hole.  Remember our projectile motion talk from the pre-lab and use 980 cm/s/s for the accel of gravity
#   - Fluid flow rate in cubic centimeters per second
# Graph and linearize the Velocity of the Fluid vs the Height to the Overflow.  When finished, answer numbers 5 through 9 in your analysis guide

# Graph and linearize the Fluid Flow Rate vs the Height to the Overflow.  When finished, answer numbers 10 through 12 in your analysis guide

# Graph and Linearize the Fluid Flor Rate vs the Velocity of the Fluid.  When finished, answer numbers 13 through 16 in your analysis guide
