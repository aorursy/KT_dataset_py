import datetime
import numpy as np
from scipy import interpolate
import calendar

#DEFINE 

## FUNCTION RETURNS AN ARRAY OF PRICES OF SIZE input4 and dates input5
## NOTE: PLEASE CHANGE THE ROUND VALUE HERE TO RETURN THE OUTPUT ARRAY ROUNDED OFF TO DESIRED PLACES (Forgot the actual question)
ROUND = 4

def interpolateFunc(input1, input2, input3, input4, input5):
    """
    input1: int
    input2: string[]
    input3: double[]
    input4: int
    input5: string[]
    
    Returns: double[] (of size input4 and rounded off to ROUND decimal places)
    """
    
    #Try to create anchor points which are grouped according to the month they fall in
    #Create a dict of anchor_groups where keys would be month
    #values would be lists of x and y values
    #Convention for x = days elapsed bw a date and startDay
    #Convention for y = price at a given date
    
    anchor_groups = {}
    
    
    # We're saving the first day of the input data as startDay
    startDay = datetime.datetime.strptime(input2[0], "%Y-%m-%d")
    
    #Finding anchor points (where derivative is zero)
    
    for i in range(1, input1-1):
        #read each date as an appropriate datetime format
        currDate = datetime.datetime.strptime(input2[i], "%Y-%m-%d")

        # Find if the price is having zero differential at this date
        if(input3[i] < input3[i+1] and input3[i] < input3[i-1]):
            currDateDiff = 1
        if(input3[i] > input3[i+1] and input3[i] > input3[i-1]):
            currDateDiff = 1
        else:
            currDateDiff = 0
        
        # If price derivative is zero at this date, add this to anchor_group
        #which will be required for fitting a cubic spline model

        if(currDateDiff):
            currMonth = currDate.month
            # If month not added in anchor_groups, create one
            if(currMonth not in anchor_groups.keys()):
                anchor_groups[currMonth] = {'X': [], 'Y': []}
            
            X_val = (currDate - startDay).days
            Y_val = input3[i]
            anchor_groups[currMonth]['X'].append(X_val)
            anchor_groups[currMonth]['Y'].append(Y_val)
            
    
    # Now we have completed the creation of anchor points
    
    #We create a final list X and Y to consist of all anchor + dummy points
    
    final_X = []
    final_Y = []
    
    for month in anchor_groups.keys():
        
        #Fit a interp1d to X and Y values of this month group
        x = np.array(anchor_groups[month]['X'])
        y = np.array(anchor_groups[month]['Y'])
        
        anchor_func = interpolate.interp1d(x, y, fill_value='extrapolate')
        
        # New dummy point for the same month in 2025
        last_day_of_month = calendar.monthrange(2025,month)[1]
        newDate = datetime.datetime(2025, month, last_day_of_month)
        new_x = (newDate-startDay).days
        
        #Predict a new anchor point
        new_y = anchor_func(new_x)
        
        #Append all the values to the final_X and final_Y
        final_X.extend(anchor_groups[month]['X'])
        final_Y.extend(anchor_groups[month]['Y'])
        
        final_X.append(new_x)
        final_Y.append(new_y)
        
    
    # We have finally got our final_X and final_Y
    # We fit a cubic spline to this superset
    
    cubic_func = interpolate.interp1d(np.array(final_X), np.array(final_Y), kind='cubic', fill_value='extrapolate')
    
    
    #Now create an output array of size input4
    # We iterate through all points in input5
    
    output_arr = []
    for i in range(input4):
        newDate = datetime.datetime.strptime(input5[i], "%Y-%m-%d")
        new_x = (newDate - startDay).days
        new_y = cubic_func(new_x)
        output_arr.append(np.round(new_y, ROUND))
        
    return np.array(output_arr)
import pandas as pd
filename = "/kaggle/input/jpmc-quant-2020/JPMC_2020_Data.csv"
df = pd.read_csv(filename)
df.head()
input1 = df.shape[0]
df['Date'] = pd.to_datetime(df['Date'])
input2 = df['Date'].dt.strftime('%Y-%m-%d').values
input3 = df['Price'].values
input4 = 7
input5 = ['2021-4-30', '2023-4-30', '2025-4-30', '2021-1-31', '2024-1-31', '2025-4-30', '2024-6-30']
interpolateFunc(input1, input2, input3, input4, input5)
y = interpolateFunc(input1, input2, input3, input1, input2)
import matplotlib.pyplot as plt
plt.plot(df['Date'].values, y, '-')
plt.plot(df['Date'].values, df['Price'].values, 'o');
np.sqrt(((y-df['Price'].values)**2).mean())
