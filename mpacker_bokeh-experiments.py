# Imports

import pandas as pd

from bokeh.plotting import figure



# Import output_file and show from bokeh.io

from bokeh.io import output_file, show, output_notebook



#importing GapMinder data from CSV stored locally on Kaggle

csvdata = pd.read_csv('../input/fertility.csv', header=0)
#creating dataframe and assigning columns 

df = pd.DataFrame(csvdata)

fertility = df['fertility']

female_literacy = df['female literacy']



output_notebook()



# Create the figure: p

p = figure(x_axis_label='fertility (children per woman)', y_axis_label ='female_literacy (% population)')



# Add a circle glyph to the figure p

p.circle(fertility, female_literacy)



# Call the output_file() function and specify the name of the file

#output_file('fert_lit.html')



# Display the plot

show(p)







#Filtering out the fertility and female literacy columns for Latin America

fertility_latinamerica = df[df['Continent']=='LAT']['fertility']

female_literacy_latinamerica = df[df['Continent']=='LAT']['female literacy']



#Filtering out the fertility and female literacy columns for Africa 

fertility_africa = df[df['Continent']=='AF']['fertility']

female_literacy_africa = df[df['Continent']=='AF']['female literacy']



#for debugging.

#print(fertility_latinamerica.head())

#print(female_literacy_latinamerica.head())

# Create the figure: Q

Q = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')



# Add a circle glyph to the figure Q

Q.circle(fertility_latinamerica, female_literacy_latinamerica)



# Add an x glyph to the figure Q

Q.x(fertility_africa,female_literacy_africa)



# Display the plot

show(Q)
# Create the figure: pk

pk = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')



# Add a blue circle glyph to the figure pk

pk.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue',size=10,alpha=0.8)



# Add a red circle glyph to the figure pk

pk.circle(fertility_africa, female_literacy_africa, color='red',size=10,alpha=0.8)







# Display the plot

show(pk)
# Imports

import pandas as pd

from bokeh.plotting import figure



# Import output_file and show from bokeh.io

from bokeh.io import output_file, show , output_notebook



#loading data

csvdata = pd.read_csv('../input/msft-adj-close/MSFT_AdjClose.csv',header = 0)

df = pd.DataFrame(csvdata)

dateColumn = df['Date']

price = df['Adj Close']



pandasDate = pd.to_datetime(dateColumn)



output_notebook()



p = figure(x_axis_type='datetime',x_axis_label=' x axis', y_axis_label='y axis')





p.line(pandasDate,price)

p.circle(pandasDate, price, fill_color="white", size=4)

 

show(p)
