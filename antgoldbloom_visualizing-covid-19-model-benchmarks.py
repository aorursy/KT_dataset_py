

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.express as px

import plotly.graph_objects as go

import plotly.subplots as subplots



import plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)  



from datetime import datetime, timedelta



line_color = ['#96508e','#4fcb93','#f86e35','#20beff','#ddaa18']

fiftyone_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California','Colorado', 'Connecticut', 'Delaware', 'District of Columbia','Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana','Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland','Massachusetts', 'Michigan', 'Minnesota', 'Mississippi','Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire','New Jersey', 'New Mexico', 'New York', 'North Carolina','North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania','Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee','Texas', 'Utah', 'Vermont', 'Virginia', 'Washington','West Virginia', 'Wisconsin', 'Wyoming']





def evaluate_models(model_list,df_benchmark_panel,common_51=True,metric='RMSLE'):



    df_benchmarks = pd.DataFrame(columns=[model_list],index=pd.date_range(start='2/1/2020', end='12/31/2020'))



        

    for model in model_list:

        for col_name in [m for m in df_benchmark_panel.columns if model in m]:

            

            forecast_criteria = ((df_benchmark_panel[col_name].notna()) & (df_benchmark_panel['Date'] > col_name[-10:]) & (datetime.strptime(col_name[-10:],'%Y-%m-%d') >= (pd.DatetimeIndex(df_benchmark_panel['Date']) - pd.DateOffset(28))))

        

            if common_51 == True:

                fiftyone_states_mask = (df_benchmark_panel['Location_Lowest_Level'].isin(fiftyone_states))

                forecast_criteria = forecast_criteria & fiftyone_states_mask

    

                       

            

            df_eval = df_benchmark_panel[forecast_criteria] 

            

            mean_population = df_eval['Population'].unique().mean()

            

            if metric == 'RMSLE':

                score = np.sqrt(((np.log(1+df_eval[col_name])-np.log(1+df_eval["Fatalities"]))**2).mean())

            elif metric == 'RMSE_pop_norm':

                score = np.sqrt( ( mean_population*((df_eval[col_name]-(df_eval["Fatalities"]))/df_eval['Population'])**2).mean()) 

                

            df_benchmarks.loc[col_name[-10:],model] = np.round(score,3)

            df_benchmarks.loc[col_name[-10:],'{}_forecasts'.format(model)] = int(len(df_eval))

    

    df_benchmarks = df_benchmarks.dropna(how='all')

    return df_benchmarks



def order_state_list(fiftyone_states):

    latest_observation = df_benchmark_panel[df_benchmark_panel['Fatalities'].notnull()]['Date'].max()

    fiftyone_states = df_benchmark_panel[(df_benchmark_panel['Date'] == latest_observation) & (df_benchmark_panel['Province_State'].isin(fiftyone_states) ) ][['Province_State','Fatalities']].sort_values(by='Fatalities',ascending=False)['Province_State'].to_list()

    return fiftyone_states



df_benchmark_panel = pd.read_csv('/kaggle/input/covid19-benchmark-panels/benchmark_panel.csv',index_col=0,low_memory=False)



us_models = ['ihme','lanl','cu80','kaggle_previous_winner','cu60','cu_nointer']



df_us_benchmark_rmsle = evaluate_models(us_models,df_benchmark_panel,True,'RMSLE')

df_us_benchmark_rmsle = df_us_benchmark_rmsle[df_us_benchmark_rmsle.index >= '2020-04-01']



df_us_benchmark_rmse_pop_norm = evaluate_models(us_models,df_benchmark_panel,True,'RMSE_pop_norm')

df_us_benchmark_rmse_pop_norm = df_us_benchmark_rmse_pop_norm[df_us_benchmark_rmse_pop_norm.index >= '2020-04-01']



fiftyone_states = order_state_list(fiftyone_states)



fig = go.Figure()



fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmse_pop_norm.index.astype(str)), y=np.array(df_us_benchmark_rmse_pop_norm[['ihme']]).flatten(),connectgaps=True,visible=True,mode='lines+markers',name='IHME',line=dict(color=line_color[1], width=3)))

fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmse_pop_norm.index.astype(str)), y=np.array(df_us_benchmark_rmse_pop_norm[['lanl']]).flatten(),connectgaps=True,visible=True,mode='lines+markers',name='LANL',line=dict(color=line_color[2], width=3)))

fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmse_pop_norm.index.astype(str)), y=np.array(df_us_benchmark_rmse_pop_norm[['cu80']]).flatten(),connectgaps=True,visible=True,mode='lines+markers',name='CU 80',line=dict(color=line_color[3], width=3)))

fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmse_pop_norm.index.astype(str)), y=np.array(df_us_benchmark_rmse_pop_norm[['kaggle_previous_winner']]).flatten(),connectgaps=True,visible=True,mode='lines+markers',name='Kaggle Leader',line=dict(color=line_color[4], width=3)))





fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmsle.index.astype(str)), y=np.array(df_us_benchmark_rmsle[['ihme']]).flatten(),connectgaps=True, visible=False, mode='lines+markers',name='IHME',line=dict(color=line_color[1], width=3)))

fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmsle.index.astype(str)), y=np.array(df_us_benchmark_rmsle[['lanl']]).flatten(),connectgaps=True, visible=False, mode='lines+markers',name='LANL',line=dict(color=line_color[2], width=3)))

fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmsle.index.astype(str)), y=np.array(df_us_benchmark_rmsle[['cu80']]).flatten(),connectgaps=True,visible=False, mode='lines+markers',name='CU 80',line=dict(color=line_color[3], width=3)))

fig.add_trace(go.Scatter(x=np.array(df_us_benchmark_rmsle.index.astype(str)), y=np.array(df_us_benchmark_rmsle[['kaggle_previous_winner']]).flatten(),connectgaps=True, visible=False, mode='lines+markers',name='Kaggle Previous Week Winner',line=dict(color=line_color[4], width=3)))







#fig['layout']['yaxis'].update(title='RMSLE')

fig['layout']['xaxis'].update(title='Forecast Date')



fig.update_layout(autosize=False,width=760,height=500,title="Benchmarking Model Performance",annotations=[dict(x=-.014,y=1.12,text='State level forecasts covering 51 US states; forecast evaluated over subsequent 29 days',showarrow = False,xref='paper', yref='paper')],

                 

                    updatemenus = [

                        go.layout.Updatemenu(

                                direction = "down", showactive=True, x = 1.13, y = 1.23,

                                buttons = list([

                                    dict(

                                        label = "RMSE normalized for population", method = "update",

                                        #args = [{"visible": [True, True, True, False, False, False,]}]                                       

                                        args = [{"visible": [True, True, True, True, False, False, False, False,]}]

                                    ),

                                    dict(

                                        label = "RMSLE", method = "update",

                                        args = [{"visible": [False, False, False, False, True, True, True, True]}]

                                        #args = [{"visible": [False, False, False, True, True, True]}]



                                    )

                                ]

                            )

                        )

                    ]



                 )



py.offline.iplot(fig)





model_display_list = us_models[0:3] #can only be four models

model_dict = dict()



for model in model_display_list:

    



    model_dict[model] = dict()

        

    

    for column in df_benchmark_panel.columns:

        if model == column[:len(model)]:

            if (model_dict[model] == {}):

                model_dict[model]['date'] = datetime.strptime(column[-10:],"%Y-%m-%d")

                model_dict[model]['date_str'] = column[-10:]

            elif (model_dict[model]['date'] < datetime.strptime(column[-10:],"%Y-%m-%d")):

                model_dict[model]['date'] = datetime.strptime(column[-10:],"%Y-%m-%d")

                model_dict[model]['date_str'] = column[-10:]





fig = go.Figure()



df_us_panel = df_benchmark_panel[df_benchmark_panel['Province_State'].isin(fiftyone_states)]

df_us_panel = df_us_panel.groupby('Date').sum()

df_us_panel[df_us_panel == 0] = None







mask = (pd.to_datetime(df_us_panel.index) > datetime.strptime('2020-03-15','%Y-%m-%d'))



visible_flag=True



i=0

for model in model_dict:

    i +=1

    fig.add_trace(go.Scatter(x=np.array(df_us_panel[mask].index.astype(str)), y=np.array(df_us_panel[mask]['{}_{}'.format(model,model_dict[model]['date_str'])]),connectgaps=True,visible=visible_flag,mode='lines',name=model,line=dict(color=line_color[i], width=3)))

    

fig.add_trace(go.Scatter(x=np.array(df_us_panel[mask].index.astype(str)), y=np.array(df_us_panel[mask]['Fatalities']),connectgaps=True,visible=visible_flag,mode='lines',name='Fatalities (Actual)',line=dict(color=line_color[0], width=3)))



fig.update_layout(title='US forecasts',autosize=False,width=760,height=500,)



py.offline.iplot(fig)



    
fig = go.Figure()



default_state = 'New York'



button_list = []





for state in fiftyone_states:



    mask = (pd.to_datetime(df_benchmark_panel['Date']) > datetime.strptime('2020-03-15','%Y-%m-%d')) & (df_benchmark_panel['Province_State'] == state)



    if state == default_state:

        visible_flag = True

    else:

        visible_flag = False

              

    i = 0

    for model in model_dict:

        i +=1

        fig.add_trace(go.Scatter(x=np.array(df_benchmark_panel[mask]['Date'].astype(str)), y=np.array(df_benchmark_panel[mask]['{}_{}'.format(model,model_dict[model]['date_str'])]),connectgaps=True,visible=visible_flag,mode='lines',name=model,line=dict(color=line_color[i], width=3)))

    

    fig.add_trace(go.Scatter(x=np.array(df_benchmark_panel[mask]['Date'].astype(str)), y=np.array(df_benchmark_panel[mask]['Fatalities']),connectgaps=True,visible=visible_flag,mode='lines',name='Fatalities (Actual)',line=dict(color=line_color[0], width=3)))



    button_list.append(dict(label = state, method = "update",args = [{"visible": np.array([[state==s]*4 for s in fiftyone_states]).flatten()}]))



fig.update_layout(autosize=False,width=760,height=500,updatemenus=[{"buttons": button_list, "direction": "down", "active": fiftyone_states.index(default_state), "showactive": True, "x": 0.3, "y": 1.12}],title="Latest State Forecasts")



py.offline.iplot(fig)


