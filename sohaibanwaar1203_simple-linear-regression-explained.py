import pandas as pd



df = pd.DataFrame({

    "Order_id" : [i for i in range(1,8)],

    "TIP($)"   : [5, 7, 11, 12, 1, 5, 11]

})



import plotly.express as px



fig = px.line(df, x="Order_id", y="TIP($)", title='Tip Calulation')

fig.show()




import plotly.graph_objects as go



# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=list(df["TIP($)"].values),

                    mode='lines+markers',

                    name='TIP'))

fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=[(sum(df["TIP($)"].values)/ len(df["TIP($)"].values)) for i in range(0,len(df["TIP($)"].values))],

                    mode='lines',

                    name='Mean'))



fig.update_layout(title='Relation in Tips and Mean')

                   

fig.show()







df["distance"]= df["TIP($)"].apply(lambda x: x - 6.8)

print(f"Error of Best Fit Line :{sum(df['distance'].values)}")
df = pd.DataFrame({

    "Order_id" : [i for i in range(1,8)],

    "TIP($)"   : [5, 7, 11, 12, 1, 5, 11],

    "Bill"   : [50, 56.5 ,100, 110, 5, 52, 101]

})





import plotly.graph_objects as go



# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=list(df["TIP($)"].values),

                    mode='lines+markers',

                    name='TIP'))

fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=list(df.Bill.values),

                    mode='lines',

                    name='Mean'))



fig.update_layout(title='Relation in Tips and Mean')

                   

fig.show()



import numpy as np 

import matplotlib.pyplot as plt 

  

def estimate_coef(x, y): 

    # number of observations/points 

    n = np.size(x) 

  

    # mean of x and y vector 

    m_x, m_y = np.mean(x), np.mean(y) 

  

    # calculating cross-deviation and deviation about x 

    SS_xy = np.sum(y*x) - n*m_y*m_x     # ss  stands for sum squared

    SS_xx = np.sum(x*x) - n*m_x*m_x 

  

    # calculating regression coefficients 

    b_1 = SS_xy / SS_xx 

    b_0 = m_y - b_1*m_x 

  

    return(b_0, b_1) 

  

def plot_regression_line(x, y, b): 

    # plotting the actual points as scatter plot 

    plt.scatter(x, y, color = "m", 

               marker = "o", s = 30) 

  

    # predicted response vector 

    y_pred = b[0] + b[1]*x 

    

    # plotting the regression line 

    plt.plot(x, y_pred, color = "g") 

  

    # putting labels 

    plt.xlabel('x') 

    plt.ylabel('y') 

  

    # function to show plot 

    plt.show()

    



    

def Linear_regression(x,y):

        # estimating coefficients 

        

        b = estimate_coef(x, y) 

        # Here b[0] is error and b[1] is the value which we got when we multiply it with x (amount bill) to get the amount of tip

        

        print("Estimated coefficients:\nb_0 = {} nb_1 = {}".format(b[0], b[1])) 

        print(b)



        # plotting regression line 

        plot_regression_line(x, y, b) 

        

Linear_regression(df["Bill"],df["TIP($)"])