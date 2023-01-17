import pandas as pd

titanic = {'Age' : [3,5,4,4,5,6,3],'Fare' : [2,10,12,16,15,5,14],'Pclass':[1,2,3,1,2,1,3]}

df = pd.DataFrame(titanic)

# print(df)



import matplotlib.pyplot as plt

scatter2 = plt.scatter(df['Age'], df['Fare'], marker = "*", c=df['Pclass'] )



# Always label your plot for clarification to the audience

plt.xlabel("Age")

plt.ylabel("Fare")

plt.title("Passenger class")



# Label the classes for the legend

classes = ["1st","2nd","3rd"]



# Draw the legend

plt.legend(handles=scatter2.legend_elements()[0], labels=classes,loc="upper right")

# Show the plot

plt.show()