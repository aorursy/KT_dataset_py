import pandas as pd                      #For importing our datasets

import matplotlib.pyplot as plt          #To visualize our data

import numpy as np                       #To do an array operation

from sklearn import linear_model as lm   #To build the linear regression model 

plt.style.use("ggplot")                  #To set the style as "ggplot" in R
df = pd.read_excel("../input/indonesian-covid19-data-7-july-2020/data covid.xlsx")



df.head()
dacov = df.loc[:, "Province_name" : "Populations"]

dacov.head()
print("The number of data sets :", df.shape[0])



for i in range(5):

    

    x = dacov.iloc[:,i].notnull().value_counts()   

    

    print(x)
dacov.describe()
dacov.sort_values(by = "Death_cases", ascending = False).head()
#dacov = np.log(dacov.loc[:, "Confirmed_cases":"Populations"])
plt.figure(figsize = (30, 20))

plt.grid(linestyle = '--', linewidth = 2)



#To visualize the whole countries



plt.scatter(x = dacov["Confirmed_cases"], y = dacov["Death_cases"], s = dacov["Populations"]/5, 

            alpha = 0.6, c = 'b')



#To only visualize top five-country



top_five = dacov.sort_values(by = "Death_cases", ascending = False).head()



plt.scatter(x = top_five["Confirmed_cases"], y = top_five["Death_cases"], 

            s = top_five["Populations"]/5, alpha = 0.6)



plt.title("Total Confirmed Cases against Total Patients Died in Indonesia", size = 45, pad =35) #pad works as the space between the title and graph

plt.xlabel("Total Confirmed Cases", size = 35, labelpad = 30) #labelpad works as the space between the label and graph

plt.ylabel("Total Patients Died", size = 35, labelpad = 30) 



plt.xticks(size = 23)

plt.yticks(size = 23)





plt.annotate("Jawa Timur", xy = (12000, 1020), size = 40, color = 'black')

plt.annotate("DKI Jakarta", xy = (12435, 649), size = 40)

plt.annotate("Kalimantan \nSelantan", xy = (2200, 200), size = 35)

plt.annotate("Jawa Tengah", xy = (4611, 230), size = 35)

plt.annotate("Sulawesi Selatan", xy = (5890, 150), size = 35)

regress = lm.LinearRegression()



x_var = np.array(dacov["Confirmed_cases"]) 

y_var = np.array(dacov["Death_cases"])



x_var = x_var.reshape(-1, 1)

y_var = y_var.reshape(-1, 1)



regress.fit(x_var, y_var)



print("Intercept = ", regress.intercept_)

print("Coefficient = ", regress.coef_)



y_predict = regress.predict(x_var)
plt.figure(figsize = (30, 20))

plt.grid(linestyle = '--', linewidth = 2)



#To visualize the whole countries



plt.scatter(x = dacov["Confirmed_cases"], y = dacov["Death_cases"], s = dacov["Populations"]/5, 

            alpha = 0.6, c = 'b')



#To only visualize top five-country



top_five = dacov.sort_values(by = "Death_cases", ascending = False).head()



plt.scatter(x = top_five["Confirmed_cases"], y = top_five["Death_cases"], 

            s = top_five["Populations"]/5, alpha = 0.6)



plt.title("Total Confirmed Cases against Total Patients Died in Indonesia", size = 45, pad =35)  #pad works as the space between the title and graph

plt.xlabel("Total Confirmed Cases", size = 35, labelpad = 30) #labelpad works as the space between the label and graph

plt.ylabel("Total Patients Died", size = 35, labelpad = 30)   



plt.xticks(size = 23)

plt.yticks(size = 23)





plt.annotate("Jawa Timur", xy = (12000, 1020), size = 40, color = 'black')

plt.annotate("DKI Jakarta", xy = (12435, 649), size = 40)

plt.annotate("Kalimantan \nSelantan", xy = (2200, 200), size = 35)

plt.annotate("Jawa Tengah", xy = (4611, 230), size = 35)

plt.annotate("Sulawesi Selatan", xy = (5890, 150), size = 35)



plt.annotate('y = -22.004+0.061X', xy = (2000, 800), size = 40)



plt.plot(x_var, y_predict, linewidth = 3)