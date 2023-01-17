# Load the relevant libraries
import pandas as pd 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import itertools
import copy
import matplotlib.lines as mlines
# Route -> TFS - FUE 
# Expected Marginal Seat Revenue
# All the demand curves follow a normal distribution

# Create dataframe to calculate probabilities, expected revenues and protection levels
# Route -> TFS - FUE 
# Expected Marginal Seat Revenue
# All the demand curves follow a normal distribution

def generateDataFrame(seatCapacity, numberOfClasses, fares, distributionParameters):
    # Create dataframe to calculate probabilities, expected revenues and protection levels
    df = pd.DataFrame(columns=['demand'])

    # form the demand list and maxrevenue list
    capacity = []
    maxRevenue = []
    for i in range(1, seatCapacity+1):
        capacity.append(i)
        maxRevenue.append(0.0)

    # Assign the capacity vecotr as demand
    df['demand'] = capacity
    df['MaxRevenue'] = maxRevenue

    for i in range(0,numberOfClasses-1):
        probVector = []
        revenueVector = []
        meanVal = distributionParameters[i][0]
        stdDev = distributionParameters[i][1]
        fare = fares[i]
        for j in range(0,seatCapacity):
            distNor = sp.norm(meanVal, stdDev)
            prob = 1- distNor.cdf(df['demand'][j])
            probVector.append(prob)
            revenueVector.append(prob*fare)
        cl1_name = "P(d >= demand|Class " + str(i+1) + ")"
        cl2_name = "Revenue_Class" + str(i+1)
        df[cl1_name] = probVector;
        df[cl2_name] = revenueVector;

    return df

# function to determine the protection levels
def getProtectionLevels(df, fares, class1_num, class2_num):
    cl1_name = "Revenue_Class" + str(class1_num)
    cl2_name = "Revenue_Class" + str(class2_num)

    pLevel = -1
    for j in range(0, seatCapacity-1):
        comFare = fares[class2_num-1]
        if (comFare < df[cl1_name][j] and comFare >= df[cl1_name][j+1]):
            pLevel = j;
            break;

    return j+1;

def plotRevenueCurves(df, numberOfclasses, show):
    # Plot the expected revenues
    plt.gcf().clear()
    for i in range(1, numberOfclasses):
        cln_name = "Revenue_Class" + str(i)
        plt.plot(df['demand'], df[cln_name], label='Class ' + str(i))
    
    if (show == True):
        plt.legend()
        plt.show()
        
    return "Plots successfully generated!"


def getAllProtectionLevels(df, fares, numberOfClasses):
    class_list = []
    pl_list = []
    my_dict = dict()
    # create list
    for i in range(1, numberOfClasses + 1):
        class_list.append(i)
        
    combination_list = list(itertools.combinations(class_list, 2))
    #print("Generated Combinations list is " + str(combination_list))
    for i in range(len(combination_list)):
        value = getProtectionLevels(df, fares, combination_list[i][0], combination_list[i][1])
        #print("The protection level between Class " + str(combination_list[i][0]) + " and Class " + str(combination_list[i][1]) 
              #+ " is " + str(value))
            
        if (combination_list[i][0] not in my_dict):
            key = combination_list[i][0]
            my_dict[key] = value
        else:
            key = combination_list[i][0]
            my_dict[key] = my_dict[key] + value
            
    #print("Protection limits dictionary : " + str(my_dict))
    available = seatCapacity
    for i in range(1, numberOfClasses):
        val = my_dict[i]
        if (available - val > 0):
            #print(available - val)
            available = available - val
            pl_list.append(val)
        else:
            pl_list.append(available)
            available = 0
    
    return pl_list
        

def getDistributionParamters(fares, class1_fare, class1_mean, std_dev):
    return_array = []
    for i in range(0, len(fares)-1):
        param  = (class1_fare/fares[i])*class1_mean, std_dev
        return_array.append(param)
    return return_array

def getTotalRevenue(fares, protection_limits, seatCapacity):
    totalRevenue = 0
    for i in range(0, len(fares)-1):
        totalRevenue += fares[i]* protection_limits[i]
        #print("multiplying " + str(fares[i]) + " and " + str(protection_limits[i]))

    if (seatCapacity - sum(protection_limits) > 0):
        #print("multiplying " + str(fares[-1]) + " and " + str(seatCapacity - sum(protection_limits)))
        totalRevenue += fares[-1]*(seatCapacity - sum(protection_limits))
    return totalRevenue

def plotEMSRCurve(fares, allocated_seats):
    # Plot base fare
    plt.plot([0, seatCapacity], [fares[-1], fares[-1]], label = "Class " + str(len(fares)))
    start = 0
    for i in range(0, len(fares)-1):
        end = start + allocated_seats[i]
        plt.plot([start, end], [fares[i], fares[i]], color = "k")
        start = end
        plt.vlines(x=end, linestyle='--', ymax=fares[i], ymin=0, colors="k")
    
    plt.plot([start, seatCapacity], [fares[-1], fares[-1]], color = "k")
    plt.vlines(x=seatCapacity, linestyle='--', ymax=fares[-1], ymin=0, colors="k")
    
    
def generateStaticAnalysis(seatCapacity, numOfClasses, faresArray, class1_fare, class1_mean, class1_std):
    print("Generating static analysis...")
    distributionParameters = getDistributionParamters(faresArray, class1_fare, class1_mean, class1_std)
    data_frame = generateDataFrame(seatCapacity, numOfClasses, faresArray, distributionParameters)
    plotRevenueCurves(data_frame, numOfClasses, False)
    pl_list = getAllProtectionLevels(data_frame, faresArray, numOfClasses)
    pl_full_list = copy.deepcopy(pl_list)
    if (seatCapacity - sum(pl_list) > 0):
        pl_full_list.append(seatCapacity - sum(pl_list))
    else:
        pl_full_list.append(0)
    print("Protection limit array for 100 seats:" + str(pl_full_list))
    revenue = getTotalRevenue(faresArray, pl_list, seatCapacity)
    print("Total revenue :" + str(revenue))
    #print(pl_full_list)
    plotEMSRCurve(faresArray, pl_full_list)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.show()
    return data_frame, revenue
    
# Question 1
# Standard paramters
seatCapacity = 100
numOfClasses = 4

# Route 1: Tenerife - Lanzarote
fares_route1 = [500, 420, 290, 125]
class1_fare_r1 = 500
class1_mean_r1 = 16.5
class1_std_r1 = 4.6
baseline_results_r1 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route1, class1_fare_r1, class1_mean_r1, class1_std_r1)
baseline_r1_data_frame = baseline_results_r1[0]
baseline_r1_revenue = baseline_results_r1[1]
print("\n")

# Route 2: Tenerife - Fuerteventura
fares_route2 = [500, 290, 250, 205]
class1_fare_r2 = 500
class1_mean_r2 = 8.8
class1_std_r2 = 3.7
baseline_results_r2 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route2, class1_fare_r2, class1_mean_r2, class1_std_r2)
baseline_r2_data_frame = baseline_results_r2[0]
baseline_r2_revenue = baseline_results_r2[1]
print("\n")

# Route 3: Tenerife - Gran Canaria
fares_route3 = [100, 70, 50, 35]
class1_fare_r3 = 100
class1_mean_r3 = 5.5
class1_std_r3 = 2.8
baseline_results_r3 =  generateStaticAnalysis(seatCapacity, numOfClasses, fares_route3, class1_fare_r3, class1_mean_r3, class1_std_r3)
baseline_r3_data_frame = baseline_results_r3[0]
baseline_r3_revenue = baseline_results_r3[1]
print("\n")
# Question 2
# Parameters
seatCapacity = 100

#------------------------------------
# Route 1: Tenerife - Lanzarote
# Attempt with 6 and 8 classes

print("----------------------\nSimulation of 6 different fare classes for Route 1: Tenerife - Lanzarote")
# 6 classes
numOfClasses = 6
fares_route1 = [950, 650, 500, 420, 290, 125]
class1_fare_r1 = 500
class1_mean_r1 = 16.5
class1_std_r1 = 4.6
r1_results_6 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route1, class1_fare_r1, class1_mean_r1, class1_std_r1)
r1_data_frame_6 = r1_results_6[0]
r1_revenue_6 = r1_results_6[1]
delta_1 =(r1_revenue_6 - baseline_r1_revenue)*100/baseline_r1_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

print("\nSimulation of 8 different fare classes for Route 1: Tenerife - Lanzarote")
# 8 classes
numOfClasses = 8
fares_route1 = [950, 650, 500, 420, 290, 230, 190, 125]
class1_fare_r1 = 500
class1_mean_r1 = 16.5
class1_std_r1 = 4.6
r1_results_8 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route1, class1_fare_r1, class1_mean_r1, class1_std_r1)
r1_data_frame_8 = r1_results_8[0]
r1_revenue_8 = r1_results_8[1]
delta_1 =(r1_revenue_8 - baseline_r1_revenue)*100/baseline_r1_revenue
delta_2 =(r1_revenue_8 - r1_revenue_6)*100/r1_revenue_6
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")
print("Increase in revenue compared to the 6 class model = " + str(round(delta_2, 2)) + " %")

#-----------------------------------
# Route 2: Tenerife - Fuerteventura
# Attempt with 6 and 8 classes

print("----------------------\nSimulation of 6 different fare classes for Route 2: Tenerife - Fuerteventura")
# 6 classes
numOfClasses = 6
fares_route2 = [620, 550, 500, 290, 250, 205]
class1_fare_r2 = 500
class1_mean_r2 = 8.8
class1_std_r2 = 3.7
r2_results_6 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route2, class1_fare_r2, class1_mean_r2, class1_std_r2)
r2_data_frame_6 = r2_results_6[0]
r2_revenue_6 = r2_results_6[1]
delta_1 =(r2_revenue_6 - baseline_r2_revenue)*100/baseline_r2_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

print("\nSimulation of 7 different fare classes for Route 2: Tenerife - Fuerteventura")
# 7 classes
numOfClasses = 7
fares_route2 = [700, 620, 550, 500, 290, 250, 205]
class1_fare_r2 = 500
class1_mean_r2 = 8.8
class1_std_r2 = 3.7
r2_results_8 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route2, class1_fare_r2, class1_mean_r2, class1_std_r2)
r2_data_frame_8 = r2_results_8[0]
r2_revenue_8 = r2_results_8[1]
delta_1 =(r2_revenue_8 - baseline_r2_revenue)*100/baseline_r2_revenue
delta_2 =(r2_revenue_8 - r2_revenue_6)*100/r2_revenue_6
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")
print("Increase in revenue compared to the 6 class model = " + str(round(delta_2, 2)) + " %")


#----------------------------------
# Route 3: Tenerife - Gran Canaria
# Attempt with 6 and 8 classes

print("----------------------\nSimulation of 6 different fare classes for Route 3: Tenerife - Gran Canaria")
# 6 classes
numOfClasses = 6
fares_route3 = [220, 100, 70, 50, 45, 35]
class1_fare_r3 = 100
class1_mean_r3 = 5.5
class1_std_r3 = 2.8
r3_results_6 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route3, class1_fare_r3, class1_mean_r3, class1_std_r3)
r3_data_frame_6 = r3_results_6[0]
r3_revenue_6 = r3_results_6[1]
delta_1 =(r3_revenue_6 - baseline_r3_revenue)*100/baseline_r3_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

print("\nSimulation of 8 different fare classes for Route 3: Tenerife - Gran Canaria")
# 8 classes
numOfClasses = 8
fares_route3 = [400, 220, 150, 120, 90, 85, 50, 45]
class1_fare_r3 = 100
class1_mean_r3 = 5.5
class1_std_r3 = 2.8
r3_results_8 = generateStaticAnalysis(seatCapacity, numOfClasses, fares_route3, class1_fare_r3, class1_mean_r3, class1_std_r3)
r3_data_frame_8 = r3_results_8[0]
r3_revenue_8 = r3_results_8[1]
delta_1 =(r3_revenue_8 - baseline_r3_revenue)*100/baseline_r3_revenue
delta_2 =(r3_revenue_8 - r3_revenue_6)*100/r3_revenue_6
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")
print("Increase in revenue compared to the 6 class model = " + str(round(delta_2, 2)) + " %")
# Question 3
# Implementation of Gabi's idea
numOfClasses = 4

def maximizeExpectedSeatVal(df, numberOfClasses,last_class_fare):
    cln_names = []
    for i in range(1, numberOfClasses):
        cln_name = "Revenue_Class" + str(i)
        cln_names.append(cln_name)
    
    
    for i in range(0, seatCapacity):
        revenueList = [last_class_fare]
        for j in range(0, len(cln_names)):
            revenueList.append(df[cln_names[j]][i])
        #print(revenueList)

        df["MaxRevenue"][i] = max(revenueList)
    
    plotRevenueCurves(df, numberOfClasses, False)
    cln_name = "MaxRevenue"
    plt.plot(df['demand'], df[cln_name], label='Maximum Expected Revenue')
    
    plt.legend()
    plt.show()
    return df


#------------------------------------
# Route 1: Tenerife - Lanzarote
fares_route1 = [500, 420, 290, 125]
r1_nw_df = maximizeExpectedSeatVal(baseline_r1_data_frame, 4, fares_route1[-1])
plt.gcf().clear()
r1_max_ex_revenue = sum(r1_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 1: Tenerife - Lanzarote = " + str(r1_max_ex_revenue))
delta_1 =(r1_max_ex_revenue - baseline_r1_revenue)*100/baseline_r1_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")
# # Route 2: Tenerife - Fuerteventura
fares_route2 = [500, 290, 250, 205]
r2_nw_df = maximizeExpectedSeatVal(baseline_r2_data_frame, 4, fares_route2[-1])
plt.gcf().clear()
r2_max_ex_revenue = sum(r2_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 2: Tenerife - Fuerteventura = " + str(r2_max_ex_revenue))
delta_1 = (r2_max_ex_revenue - baseline_r2_revenue)*100/baseline_r2_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

# Route 3: Tenerife - Gran Canaria
fares_route3 = [100, 70, 50, 35]
r3_nw_df = maximizeExpectedSeatVal(baseline_r3_data_frame, 4, fares_route3[-1])
plt.gcf().clear()
r3_max_ex_revenue = sum(r3_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 3: Tenerife - Gran Canaria = " + str(r3_max_ex_revenue))
delta_1 = (r3_max_ex_revenue - baseline_r3_revenue)*100/baseline_r3_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")
# Question 4: Form a hybrid model according to Mike and see the results

#------------------------------------
# Route 1: Tenerife - Lanzarote
# Attempt with 6 and 8 classes

print("----------------------\nHybrid Model for Route 1 with 6 class fares: Tenerife - Lanzarote")
# 6 classes
numOfClasses = 6
fares_route1 = [500, 420, 290, 200, 170, 125]
r1_nw_df = maximizeExpectedSeatVal(r1_data_frame_6, 6, fares_route1[-1])
plt.gcf().clear()
r1_max_ex_revenue = sum(r1_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 1: Tenerife - Lanzarote = " + str(r1_max_ex_revenue))
delta_1 =(r1_max_ex_revenue - baseline_r1_revenue)*100/baseline_r1_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

print("\nHybrid Model for Route 1 with 8 class fares: Tenerife - Lanzarote")
# 8 classes
numOfClasses = 8
fares_route1 = [500, 420, 290, 200, 170, 160, 145, 125]
r1_nw_df = maximizeExpectedSeatVal(r1_data_frame_8, 8, fares_route1[-1])
plt.gcf().clear()
r1_max_ex_revenue = sum(r1_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 1: Tenerife - Lanzarote = " + str(r1_max_ex_revenue))
delta_1 =(r1_max_ex_revenue - baseline_r1_revenue)*100/baseline_r1_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

#-----------------------------------
# Route 2: Tenerife - Fuerteventura
# Attempt with 6 and 8 classes

print("----------------------\nHybrid Model for Route 2 with 6 classes: Tenerife - Fuerteventura")
# 6 classes
numOfClasses = 6
fares_route2 = [620, 550, 500, 290, 250, 205]
r2_nw_df = maximizeExpectedSeatVal(r2_data_frame_6, 6, fares_route2[-1])
plt.gcf().clear()
r2_max_ex_revenue = sum(r2_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 2: Tenerife - Fuerteventura = " + str(r2_max_ex_revenue))
delta_1 = (r2_max_ex_revenue - baseline_r2_revenue)*100/baseline_r2_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

print("\nHybrid Model for Route 2 with 7 classes: Tenerife - Fuerteventura")
# 7 classes
numOfClasses = 7
fares_route2 = [700, 620, 550, 500, 290, 250, 205]
r2_nw_df = maximizeExpectedSeatVal(r2_data_frame_8, 7, fares_route2[-1])
plt.gcf().clear()
r2_max_ex_revenue = sum(r2_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 2: Tenerife - Fuerteventura = " + str(r2_max_ex_revenue))
delta_1 = (r2_max_ex_revenue - baseline_r2_revenue)*100/baseline_r2_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

#----------------------------------
# Route 3: Tenerife - Gran Canaria
# Attempt with 6 and 8 classes

print("----------------------\nHybrid Model for Route 3 with 6 classes: Tenerife - Gran Canaria")
# 6 classes
numOfClasses = 6
fares_route3 = [220, 100, 70, 50, 45, 35]
r3_nw_df = maximizeExpectedSeatVal(r3_data_frame_6, 6, fares_route3[-1])
plt.gcf().clear()
r3_max_ex_revenue = sum(r3_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 3: Tenerife - Gran Canaria = " + str(r3_max_ex_revenue))
delta_1 = (r3_max_ex_revenue - baseline_r3_revenue)*100/baseline_r3_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

print("\nHybrid Model for Route 3 with 8 classes: Tenerife - Gran Canaria")
# 8 classes
numOfClasses = 8
fares_route3 = [400, 220, 100, 70, 50, 45, 42, 35]
r3_nw_df = maximizeExpectedSeatVal(r3_data_frame_8, 8, fares_route3[-1])
plt.gcf().clear()
r3_max_ex_revenue = sum(r3_nw_df["MaxRevenue"])
print("Maximum Expected Revenue for Route 3: Tenerife - Gran Canaria = " + str(r3_max_ex_revenue))
delta_1 = (r3_max_ex_revenue - baseline_r3_revenue)*100/baseline_r3_revenue
print("Increase in revenue compared to baseline model = " + str(round(delta_1, 2)) + " %")

