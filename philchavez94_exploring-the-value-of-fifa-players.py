import pandas as pd

data_frame = pd.read_csv('../input/fifa19/data.csv')

data_frame.shape
data_frame.describe()
data_frame[data_frame["Age"]>40]
#Creating df to find difference 



df1 = pd.DataFrame(data_frame, columns=['Name','Wage','Value'])



#Cleaning data by converting str to int



def value_to_float(x):

    if type(x) == float or type(x) == int:

        return x

    if 'K' in x:

        if len(x) > 1:

            return float(x.replace('K', '')) * 1000

        return 1000.0

    if 'M' in x:

        if len(x) > 1:

            return float(x.replace('M', '')) * 1000000

        return 1000000.0

    if 'B' in x:

        return float(x.replace('B', '')) * 1000000000

    return 0.0



#Removing € from data



wage = df1['Wage'].replace('[\€,]','',regex=True).apply(value_to_float)

value = df1['Value'].replace('[\€,]','',regex=True).apply(value_to_float)



#Using cleaned variables



df1['Wage'] = wage

df1['Value'] = value



#Creating new column with calculated values and sorting



df1['Difference']= df1['Value'] - df1['Wage']

df1.sort_values('Difference', ascending=False)
import seaborn as sns

sns.set()



graph = sns.scatterplot(x='Wage', y='Value', data=df1)

graph
#Interactive Visulazation allowing us to hover over scatterplot points



from bokeh.plotting import figure,show, output_notebook

from bokeh.models import HoverTool





TOOLTIPS = HoverTool(tooltips=[

    ("index", "$index"),

    ("(Wage,Value)", "(@Wage, @Value)"),

    ("Name", "@Name")]

                    )



p = figure(title='Value Vs Wage of Soccer Players', x_axis_label='Wage', y_axis_label='Value', plot_width=700, plot_height=700, tools=[TOOLTIPS])

p.circle('Wage', 'Value', size=10, source=df1)

output_notebook()

show(p)


