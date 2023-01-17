import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Function that calculates the Gravity Function for each point
# Due to memory issues, I was not able to store temporary data, just the result
# The gravity function of a given point is the sum of the gravity function bewtween this point and all others
# The gravity function of two points 1 and 2 equals to M1 * M2 /  (distance between theses two points ) **2

def Gravity_Function(df):
    for i in df.index:
        #print(i)
        currentx, currenty, currentm = df.loc[i,"X"], df.loc[i,"Y"], df.loc[i,"m"]
        mask1 = df.index == i
        temp = df[~(mask1)].copy()
        temp["distance"] =  temp["m"] * currentm / ( (currentx-temp["X"]) **2 + (currenty- temp["Y"])**2)
        dist = temp["distance"].sum()
        df.loc[i,"gf"] = dist
# Function that returns the gravity function of the closest point for a given point times k_lambda (between 0 and 1)
# k_lamba is the step size of the change in the gravity function of the points        
# Processes a batch of points

def Gravity_Function_batch(df,df_points, k_lambda):
        lista = []
        for i in df_points.index.values:
            currentx, currenty, current_gf = df_points.loc[i ,"X"], df_points.loc[i,"Y"], df_points.loc[i,"gf"]
            mask1 = df.index == i
            temp = df[~(mask1)].copy()
            temp["distance"] =  (currentx-temp["X"]) **2 + (currenty- temp["Y"])**2
            closest_point = temp[temp["distance"]== temp["distance"].min()]["gf"].index.values[0]
            closest_point_gf = temp[temp["distance"]== temp["distance"].min()]["gf"].values[0]
            if  closest_point_gf > current_gf :
                walk_to = current_gf + ( closest_point_gf - current_gf) * k_lambda
            else:
                walk_to = closest_point_gf + ( current_gf - closest_point_gf) * k_lambda
                
            lista = lista + [[i, closest_point, walk_to ]]
        return lista  
# Function that calculates the cost of the current solution, that is the euclidean distance of the ordered points
    
def Cost(df):
    for i in df.index:
        if i ==0 :
            df.loc[i,"Cost"] = 0
        else:
            df.loc[i,"Cost"] =  math.sqrt( (df.loc[i,"X"] - df.loc[i-1,"X"])**2  + (df.loc[i,"Y"] - df.loc[i-1,"Y"])**2)    
# Function that changes the mass of a point so that it calculates a giver gravity funcion
            
def Invert_Gravity(df,df_points):    
        for i,j in df_points:
            currentx, currenty = df.loc[i,"X"], df.loc[i,"Y"]
            mask1 = df.index == i
            temp = df[~(mask1)].copy()
            temp["distance"] =  temp["m"]  / ( (currentx-temp["X"]) **2 + (currenty- temp["Y"])**2)
            dist = temp["distance"].sum()
            new_m = j / dist
            df.loc[i,"m"] = new_m
# READ FILE ------------------------------------------------------------------
df= pd.read_csv("../input/cities.csv")
  
# Simplicity and performance -------------------------------------------------
df= df.head(200)
# INIT_STATE -----------------------------------------------------------------------------------

#INITIALIZE m, some possibilities:

df["m"] = df.index * 10
df.loc[199,"m"] = 500000

# OR
# max= df["X"].max()
# df["m"] = df["X"] * df["Y"]+ max

#OR
# max= df["X"]
# df["m"] = df["X"] * df["Y"]

k_lambda = 0.9
batch = 0.8
batch = int(df.shape[0] * batch)

cost = 1000000000000000
# MINMAX SCALER --------------------------------------------------------------------------------
# I "think" scaling would help to balance the mass and the computed distances, but I am not sure yet

scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

scalerX.fit(df["X"].values.reshape(df.shape[0],1))
df["X"]=scalerX.transform(df["X"].values.reshape(df.shape[0],1))    
    
scalerY.fit(df["Y"].values.reshape(df.shape[0],1))
df["Y"]=scalerY.transform(df["Y"].values.reshape(df.shape[0],1))  
# PROPAGATE ---------------------------------------------------------------------

for i in range(0,20):

    # ----------- FORWARD ---------------------------------------------------------
    print("Forward")
    
    # 1. Calculate the Gravitational Function for all points
    print("Calculating GF")
    Gravity_Function(df) 
       
    # 2. Order space_continuum by gravity of points
    df.sort_values(by="gf", ascending= True, inplace=True)
    plt.plot(df["X"], df["Y"])
    plt.show()
    df.reset_index(drop=True, inplace=True)
    
    # 3. Calculate the Cost Function, that is the distance between the ordered points
    print("Calculating cost...")
    Cost(df)
    this_cost =  df["Cost"].sum()
    if this_cost < cost:
        cost = this_cost
        df.to_csv("menor_custo.csv")
    print("Cost ", this_cost)
    
    
    # -------------- BACKWARD ----------------------------------------------------
    print("Backward")
    
    # 4. Now lets find the point that has the biggest distance cost, and change its m so it is
    # closer the his closest point
    # The batch can be one or several points !!!! 
    print("Selecting points")
    points= df.sort_values(by="Cost", ascending=False).head(int(batch* df.shape[0]))[["X","Y","gf"]].copy() 
    
    # 5. Now we have to go back and change m(i) in a way that i will be close to its closest point
    # lets find out what is our target gravity for this point
    # print("Point ",point["gf"].values[0], point.index.values[0])
    print("Calculating Gravity inverse")
    target_gravities = Gravity_Function_batch(df,points,k_lambda)
    df_points = np.asarray(target_gravities)[:, [0,2]] 
    #print("Closest point ", target_gravity )
    
    # 6. Now move towards the target gravity, but not completely. 
    # print("Adjusted target", target_gravity )   
    # Now lets find m that gives us this target gravity
    print("Inverting Gravity")
    Invert_Gravity(df,df_points)

