# %%

import numpy as np

import itertools

import plotly.express as px

import pandas as pd

from plotly.subplots import make_subplots

from multiprocessing import Pool





class CoronaSim:

    def __init__(self, grid_size, initial_virus, recover_time, speedreaction, incubation, virulence, contactsize=1, num_cores=4):

        self.sim_grid = np.zeros(shape=[grid_size, grid_size])

        ini_x_virus = np.random.randint(

            low=0, high=grid_size, size=initial_virus)

        ini_y_virus = np.random.randint(

            low=0, high=grid_size, size=initial_virus)

        self.inistate_matrix = np.zeros(shape=[grid_size, grid_size])

        self.inistate_matrix.fill(float(recover_time))

        self.recover_time = recover_time

        self.inistate_matrix[ini_x_virus, ini_y_virus] = 7

        self.speedreaction = speedreaction

        self.incubation = incubation

        self.samplesize = contactsize

        self.virulence = virulence

        self.num_cores = num_cores

        self.all_sites = list(itertools.product(

            range(self.sim_grid.shape[0]), range(self.sim_grid.shape[0])))



    def mechanismcheck(self):

        state_value = np.arange(31)

        valuedf = pd.DataFrame(

            {'state': state_value, 'Activity': self.activity(state_value)})

        f1 = px.scatter(valuedf, x="state", y="Activity")

        f1.data[0].update(mode='markers+lines')

        f1.update_traces(line_color='#B54434',

                         marker_line_width=3, marker_size=4)



        distance = np.arange(200)

        disp = np.exp(-self.gm_virulence(20)*distance**2)

        contactdf = pd.DataFrame({'distance': distance, 'disp': disp})

        f2 = px.line(contactdf, x="distance", y="disp")

        f2.data[0].update(mode='markers+lines')

        f2.update_traces(line_color='#1B813E',

                         marker_line_width=3, marker_size=4)



        infected_num = np.arange(10000)

        measuredf = pd.DataFrame(

            {'infected_num': infected_num, 'measure': self.gm_virulence(infected_num)})

        f3 = px.line(measuredf, x="infected_num", y="measure")

        f3.update_traces(line_color='#66327C',

                         marker_line_width=3, marker_size=4)



        trace1 = f1['data'][0]

        trace2 = f2['data'][0]

        trace3 = f3['data'][0]



        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, subplot_titles=(

            "Figure 1", "Figure 2", "Figure 3"))

        fig.add_trace(trace1, row=1, col=1)

        fig.add_trace(trace2, row=2, col=1)

        fig.add_trace(trace3, row=3, col=1)



        # Update xaxis properties

        fig.update_xaxes(title_text="Health state", row=1, col=1)

        fig.update_xaxes(title_text="Distance", range=[10, 50], row=2, col=1)

        fig.update_xaxes(title_text="The number of infected cases",

                         showgrid=False, row=3, col=1)



        # Update yaxis properties

        fig.update_yaxes(title_text="Willingness", row=1, col=1)

        fig.update_yaxes(title_text="Contact rate",

                         showgrid=False, row=2, col=1)

        fig.update_yaxes(

            title_text="Intensity of the restriction", row=3, col=1)



        # fig['layout'].update(height=800, width=800, showlegend=False)

        fig.update_layout(

            xaxis=dict(

                showline=True,

                showgrid=False,

                showticklabels=True,

                linecolor='rgb(204, 204, 204)',

                linewidth=2,

                ticks='outside',

                tickfont=dict(

                    family='Arial',

                    size=12,

                    color='rgb(82, 82, 82)',

                ),

            ),

            yaxis=dict(

                showline=True,

                showgrid=False,

                showticklabels=True,

                linecolor='rgb(204, 204, 204)',

                linewidth=2,

                ticks='outside',

                tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                ),

            ),

            xaxis2=dict(

                showline=True,

                showgrid=False,

                showticklabels=True,

                linecolor='rgb(204, 204, 204)',

                linewidth=2,

                ticks='outside',

                tickfont=dict(

                    family='Arial',

                    size=12,

                    color='rgb(82, 82, 82)',

                ),

            ),

            yaxis2=dict(

                showline=True,

                showgrid=False,

                showticklabels=True,

                linecolor='rgb(204, 204, 204)',

                linewidth=2,

                ticks='outside',

                tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                ),

            ),

            xaxis3=dict(

                showline=True,

                showgrid=False,

                showticklabels=True,

                linecolor='rgb(204, 204, 204)',

                linewidth=2,

                ticks='outside',

                tickfont=dict(

                    family='Arial',

                    size=12,

                    color='rgb(82, 82, 82)',

                ),

            ),

            yaxis3=dict(

                showline=True,

                showgrid=False,

                showticklabels=True,

                linecolor='rgb(204, 204, 204)',

                linewidth=2,

                ticks='outside',

                tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                ),

            ),

            autosize=True,



            plot_bgcolor='white',

            height=800, width=800,

        )

        fig.show()



    def activity(self, state):

        disp = np.exp((state-self.incubation) ** 2 /

                      self.virulence ** 2)

        return disp



    def gm_virulence(self, infected_num):

        return 100*(2/(1+np.exp(-infected_num*self.speedreaction/(self.sim_grid.shape[0]*self.sim_grid.shape[1])))-1)



    def spread_prob(self, x_row, y_col, state, seed=1):

        np.random.seed(seed)

        distance_sites = np.linalg.norm(

            np.array(self.all_sites) - np.array([x_row, y_col]), axis=1)

        Act = self.activity(state)

        gm_virulence = self.gm_virulence(

            infected_num=len(np.where(state < self.recover_time)[0]))

        prob_spread = np.exp(-gm_virulence *

                             distance_sites ** 2) * Act[x_row, y_col] * Act.flatten()

        prob_spread[x_row*self.sim_grid.shape[1]+y_col] = 0

        focal_state = np.random.choice(range(

            self.sim_grid.shape[0]*self.sim_grid.shape[1]), size=self.samplesize, p=prob_spread/sum(prob_spread))

        focal_state_value = 0 if min(state.flatten()[focal_state]) < self.recover_time else self.recover_time

        return focal_state_value



    def simspread(self, t_end, savefile):

        self.savefile = savefile

        state_matrix = self.inistate_matrix

        output_list = []

        parallel_cores = Pool(self.num_cores)

        for t in range(t_end):

            num_infected = len(np.where(state_matrix < self.recover_time)[0])

            print(

                f'At Day {t}, {num_infected} infected cases are confirmed...')

            healthy_individual_index_row = np.where(state_matrix >= self.recover_time)[0]

            healthy_individual_index_col = np.where(state_matrix >= self.recover_time)[1]

            change_state = parallel_cores.starmap(self.spread_prob,

                                                  zip(healthy_individual_index_row, healthy_individual_index_col, itertools.repeat(state_matrix)))

            state_matrix[healthy_individual_index_row,

                         healthy_individual_index_col] = change_state

            state_matrix += 1

            output_list.append(state_matrix.tolist())

        np.savez(self.savefile, *output_list)

        return state_matrix

    

if __name__ == "__main__":

    test = CoronaSim(grid_size=100, initial_virus=5, contactsize=2,num_cores=6,

                         recover_time=30, speedreaction=0.01, incubation=10, virulence=25)

    test.mechanismcheck()
# Start running simulations

result = test.simspread(t_end=10, savefile='test.npz')
# Simulation setup

scenario1 = CoronaSim(grid_size=100, initial_virus=5, contactsize=2, num_cores=6,

                     recover_time=30, speedreaction=0.01, incubation=7, virulence=25)
# %%

import plotly.graph_objects as go

import numpy as np

import pandas as pd



num_infected = []

Day = []

batch_list = []

for batch in range(1, 4):

    savefile = f'../input/simulation-scripts/outfile_s{batch}.npz'

    container = np.load(savefile)

    sim_result = [container[key] for key in container]

    for t in range(len(sim_result)):

        num_infected.append(len(np.where(sim_result[t] < 30)[0]))

    Day.extend(np.arange(len(sim_result)).tolist())

    batch_list.extend(np.repeat(batch, len(sim_result)))



infected_growth_df = pd.DataFrame(

    {'num_infected': num_infected, 'Day': Day, 'batch': batch_list})



# %%





# Add data



fig = go.Figure()

# Create and style traces

fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 1].Day, y=infected_growth_df[infected_growth_df['batch'] == 1].num_infected, name='Speed 0.01',

                         line=dict(color='firebrick', width=4)))

fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 2].Day, y=infected_growth_df[infected_growth_df['batch'] == 2].num_infected, name='Speed 0.1',

                         line=dict(color='royalblue', width=4,

                                   dash='dot')))

fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 3].Day, y=infected_growth_df[infected_growth_df['batch'] == 3].num_infected, name='Speed 1',

                         line=dict(color='green', width=4,

                                   dash='dash')  # dash options include 'dash', 'dot', and 'dashdot'

                         ))



# Edit the layout

fig.update_layout(title='The influence of government reaction speed on the pandemic development',

                  xaxis_title='Day',

                  yaxis_title='Number of infected cases',

                  xaxis=dict(

                        showline=True,

                        showgrid=False,

                        showticklabels=True,

                        linecolor='rgb(204, 204, 204)',

                        linewidth=2,

                        ticks='outside',

                        tickfont=dict(

                            family='Arial',

                            size=12,

                            color='rgb(82, 82, 82)',

                        ),

                  ),

                  yaxis=dict(

                      showline=True,

                      showgrid=False,

                      showticklabels=True,

                      linecolor='rgb(204, 204, 204)',

                      linewidth=2,

                      ticks='outside',

                      tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                      ),

                  ),

                  autosize=True,

                  plot_bgcolor='white',

                  height=600, width=800

                  )



fig.show()



# %%

# %%

import plotly.graph_objects as go

import numpy as np

import pandas as pd



num_infected = []

Day = []

batch_list = []

for batch in range(1, 4):

    savefile = f'../input/simulation-scripts/outfile_s{batch}.npz'

    container = np.load(savefile)

    sim_result = [container[key] for key in container]

    acc_list = []

    for t in range(1,len(sim_result)):

        acc_list.append(len(np.where(sim_result[t] < 30)[0])-len(np.where(sim_result[t-1] < 30)[0]))

    num_infected.extend(acc_list)

    Day.extend(np.arange(len(sim_result)-1).tolist())

    batch_list.extend(np.repeat(batch, len(sim_result)-1))



infected_growth_df = pd.DataFrame(

    {'num_infected': num_infected, 'Day': Day, 'batch': batch_list})



# %%





# Add data



fig = go.Figure()

# Create and style traces

fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 1].Day, y=infected_growth_df[infected_growth_df['batch'] == 1].num_infected, name='Speed 0.01',

                         line=dict(color='firebrick', width=4),fill='tozeroy'))

fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 2].Day, y=infected_growth_df[infected_growth_df['batch'] == 2].num_infected, name='Speed 0.1',

                         line=dict(color='royalblue', width=4,

                                   dash='dot'),fill='tozeroy'))

fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 3].Day, y=infected_growth_df[infected_growth_df['batch'] == 3].num_infected, name='Speed 1',

                         line=dict(color='green', width=4,

                                   dash='dash'),  # dash options include 'dash', 'dot', and 'dashdot'

                         fill='tozeroy'))



# Edit the layout

fig.update_layout(title='',

                  xaxis_title='Day',

                  yaxis_title='Number of newly increase infected cases',

                  xaxis=dict(

                        showline=True,

                        showgrid=False,

                        showticklabels=True,

                        linecolor='rgb(204, 204, 204)',

                        linewidth=2,

                        ticks='outside',

                        tickfont=dict(

                            family='Arial',

                            size=12,

                            color='rgb(82, 82, 82)',

                        ),

                  ),

                  yaxis=dict(

                      showline=True,

                      showgrid=False,

                      showticklabels=True,

                      linecolor='rgb(204, 204, 204)',

                      linewidth=2,

                      ticks='outside',

                      tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                      ),

                  ),

                  autosize=True,



                  plot_bgcolor='white',

                  height=600, width=800,

                  )



fig.show()



# %%

import plotly.express as px

import pandas as pd

import plotly.graph_objects as go

import numpy as np



datafile = '../input/covid19-global-forecasting-week-2/train.csv'

data = pd.read_csv(datafile)

data['PSCR'] = data.Province_State.map(str)+ '' + data.Country_Region.map(str)



region = pd.unique(data['PSCR']).tolist()

f_region = []

time_list = []

region_name = []

for ci in range(len(region)):

    region_data = data[data['PSCR'] == region[ci]]

    region_data = region_data[region_data.ConfirmedCases > 0]

    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(

    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()

    # Only considering the countries with effective data

    if len(np.where(inc_percentage > 0)[0]) > 0:

        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]

        f_region.extend(inc_percentage)

        time_list.extend([i for i in range(len(inc_percentage))])

        region_name.extend([region[ci] for i in range(len(inc_percentage))])

    else:

        pass

f_df = pd.DataFrame(

    {'increase': f_region, 'Day': time_list, 'region': region_name})



fig = px.line(f_df, x='Day',

              y='increase', color='region')

fig.update_layout(title='ip patterns',

                  xaxis_title='Day',

                  yaxis_title='Increasing percentage',

                  xaxis=dict(

                        showline=True,

                        showgrid=False,

                        showticklabels=True,

                        linecolor='rgb(204, 204, 204)',

                        linewidth=2,

                        ticks='outside',

                        tickfont=dict(

                            family='Arial',

                            size=12,

                            color='rgb(82, 82, 82)',

                        ),

                  ),

                  yaxis=dict(

                      showline=True,

                      showgrid=False,

                      showticklabels=True,

                      linecolor='rgb(204, 204, 204)',

                      linewidth=2,

                      ticks='outside',

                      tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                      ),

                  ),

                  autosize=True,



                  plot_bgcolor='white',

                  height=600, width=800,

                  )

fig.show()
import plotly.express as px

import pandas as pd

import numpy as np



datafile = '../input/covid19-global-forecasting-week-2/train.csv'

data = pd.read_csv(datafile)



# %%

all_region_data = data[pd.isna(data['Province_State'])]

region = ['Japan', 'Israel']

# region = pd.unique(all_region_data['Country_Region']).tolist()

f_region = []

time_list = []

region_name = []

for ci in range(len(region)):

    region_data = data[data['Country_Region'] == region[ci]]

    region_data = region_data[region_data.ConfirmedCases > 0]

    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(

    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()

    # Only considering the countries with effective data

    if len(np.where(inc_percentage > 0)[0]) > 0:

        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]

        f_region.extend(inc_percentage)

        time_list.extend([i for i in range(len(inc_percentage))])

        region_name.extend([region[ci] for i in range(len(inc_percentage))])

    else:

        pass

f_df = pd.DataFrame(

    {'increase': f_region, 'Day': time_list, 'region': region_name})





# %%

sim_data = []

speed = [0.01,0.1,1]

for batch in range(1,4):

    result = f'../input/simulation-scripts/outfile_s{batch}.npz'

    container = np.load(result)

    speed_batch = f'Sim: speed {speed[batch-1]}'



    sim_result = [container[key] for key in container]

    num_infected = []

    for t in range(len(sim_result)):

        num_infected.append(len(np.where(sim_result[t] < 30)[0]))



    inc_infected = [(num_infected[i+1]-num_infected[i])/num_infected[i]

                    for i in range(len(num_infected)-1)]

    infected_growth_df = pd.DataFrame({'increase': inc_infected, 'Day': [

        i for i in range(len(sim_result)-1)], 'region': speed_batch})

    sim_data.append(infected_growth_df)

sim_df = pd.concat(sim_data)

# %%

newf = f_df.append(sim_df)



# %%

fig = px.line(newf, x='Day',

              y='increase', color='region')

fig.update_layout(title='ip patterns of Japan and Israel against 3 simulations',

                  xaxis_title='Day',

                  yaxis_title='Increasing percentage',

                  xaxis=dict(

                        showline=True,

                        showgrid=False,

                        showticklabels=True,

                        linecolor='rgb(204, 204, 204)',

                        linewidth=2,

                        ticks='outside',

                        tickfont=dict(

                            family='Arial',

                            size=12,

                            color='rgb(82, 82, 82)',

                        ),

                  ),

                  yaxis=dict(

                      showline=True,

                      showgrid=False,

                      showticklabels=True,

                      linecolor='rgb(204, 204, 204)',

                      linewidth=2,

                      ticks='outside',

                      tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                      ),

                  ),

                  autosize=True,



                  plot_bgcolor='white',

                  height=400, width=600,

                  )



fig.show()
# %%

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

import pandas as pd





class plotresult:

    def __init__(self, savefile):

        container = np.load(savefile)

        self.sim_result = [container[key] for key in container]



    def infectiongrowth(self):

        num_infected = []

        for t in range(len(self.sim_result)):

            num_infected.append(len(np.where(self.sim_result[t] < 30)[0]))

        infected_growth_df = pd.DataFrame({'num_infected': num_infected, 'Day': [

                                          i for i in range(len(self.sim_result))]})

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=infected_growth_df.Day, y=infected_growth_df['num_infected'], name="AAPL High",

                                 line_color='deepskyblue'))



        fig.update_layout(title_text='Infection growth',

                          xaxis_rangeslider_visible=True)

        fig.show()



    def infectionheatmap(self):

        infect_dis = []

        col = []

        row = []

        days = []

        for t in range(len(self.sim_result)):

            temp_re = self.sim_result[t].tolist()

            flatten_re = [item for sublist in temp_re for item in sublist]

            x_co = np.tile(range(len(temp_re)), len(temp_re))

            y_co = np.repeat(range(len(temp_re)), len(temp_re))

            day_series = np.repeat(t, len(temp_re)**2)



            infect_dis.extend(flatten_re)

            col.extend(x_co)

            row.extend(y_co)

            days.extend(day_series)



        heatmapdf = pd.DataFrame(

            {'dis': infect_dis, 'Day': days, 'col': col, 'row': row})

        fig = px.scatter(heatmapdf, x="col", y="row", color='dis', animation_frame="Day",

                         color_continuous_scale=[(0, "#81C7D4"), (0.2, "#D0104C"), (1, "#81C7D4")])

        fig.update_layout(title='The pandemic development',

                          xaxis_title='',

                          yaxis_title='',

                          xaxis=dict(

                              showline=False,

                              showgrid=False,

                              showticklabels=False,

                          ),

                          yaxis=dict(

                              showline=False,

                              showgrid=False,

                              showticklabels=False,

                          ),

                          autosize=True,

                          plot_bgcolor='white',

                          height=600, width=600,

                          coloraxis_colorbar=dict(

                              title="Healthy state"

                          )

                          )



        fig.show()





        # %%

if __name__ == "__main__":

    result = '../input/simulation-scripts/outfile_s1.npz'

    testplot = plotresult(result)

    # testplot.infectiongrowth()

    testplot.infectionheatmap()



# %%

import plotly.express as px

import pandas as pd

import plotly.graph_objects as go

import numpy as np



datafile = '../input/covid19-global-forecasting-week-2/train.csv'

data = pd.read_csv(datafile)

data['PSCR'] = data.Province_State.map(str)+data.Country_Region.map(str)



# %%

# ip pattern of the empirical data from 2020/03/19 onwards

region = pd.unique(data['PSCR']).tolist()

f_region = []

time_list = []

region_name = []

actual_date = []

no_infection_country = []

for ci in range(len(region)):

    region_data = data[data['PSCR'] == region[ci]]

    region_data = region_data[region_data.ConfirmedCases > 0]

    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(

    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()

    # Only considering the countries with effective data

    if len(np.where(inc_percentage > 0)[0]) > 0:

        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]

        actual_date.append(region_data.Date[1:])

        f_region.extend(inc_percentage)

        time_list.extend([i for i in range(len(inc_percentage))])

        region_name.extend([region[ci] for i in range(len(inc_percentage))])

    else:

        no_infection_country.append(region[ci])

f_df = pd.DataFrame(

    {'increase': f_region, 'Day': time_list, 'PSCR': region_name})





# %%

# Simulation data for training

sim_data = []

speed = [0.01,0.1,1]

for batch in range(1,4):

    result = f'../input/simulation-scripts/outfile_s{batch}.npz'

    container = np.load(result)

    speed_batch = f'Sim: speed {speed[batch-1]}'



    sim_result = [container[key] for key in container]

    num_infected = []

    for t in range(len(sim_result)):

        num_infected.append(len(np.where(sim_result[t] < 30)[0]))



    inc_infected = [(num_infected[i+1]-num_infected[i])/num_infected[i]

                    for i in range(len(num_infected)-1)]

    infected_growth_df = pd.DataFrame({'increase': inc_infected, 'Day': [

        i for i in range(len(sim_result)-1)], 'PSCR': speed_batch})

    sim_data.append(infected_growth_df)

sim_df = pd.concat(sim_data)



# %%

criteria_day_length = 10

sim_class_ip = []

for speed in pd.unique(sim_df.PSCR):

    sim_class_ip.append(sim_df[sim_df['PSCR'] == speed].increase.tolist())

sim_class_ip_array = np.array(sim_class_ip)



#%%

labels = []

effective_region = []

for region_loop in region:

    if region_loop not in no_infection_country:

        ip = f_df[f_df['PSCR'] == region_loop].increase[:criteria_day_length].tolist()

        euclidean_dis = np.linalg.norm(np.array(ip)-sim_class_ip_array[:,:len(ip)],axis = 1)

        labels.append(np.where(euclidean_dis == min(euclidean_dis))[0][0])

        effective_region.append(region_loop)

    else:

        pass



xlabels = ['Slow','Moderate','Fast']

scenario_class = {'ip': [xlabels[i] for i in labels], 'Area':effective_region, 'width': [1 for i in range(len(labels))]}

sce_df = pd.DataFrame(scenario_class)

#%%

fig = px.bar(sce_df, x="ip", y="width", color='Area', height=400)

fig.update_layout(title='Strategies of regions',

                  xaxis_title='Strategy',

                  yaxis_title='Areas and regions',

                  xaxis=dict(

                        showline=True,

                        showgrid=False,

                        showticklabels=True,

                        linecolor='rgb(204, 204, 204)',

                        linewidth=2,

                        ticks='outside',

                        tickfont=dict(

                            family='Arial',

                            size=12,

                            color='rgb(82, 82, 82)',

                        )

                  ),

                  yaxis=dict(

                      showline=True,

                      showgrid=False,

                      showticklabels=True,

                      linecolor='rgb(204, 204, 204)',

                      linewidth=2,

                      ticks='outside',

                      tickfont=dict(

                          family='Arial',

                          size=12,

                          color='rgb(82, 82, 82)',

                      ),

                  ),

                  autosize=True,

                  plot_bgcolor='white',

                  height=600, width=800,

                  )

fig.show()
# Using the data on 18 Mar to calculate the tendency of the pandemic.

date_datause = '2020-03-18'

date_actualdata = '2020-03-30'

date_length = (pd.to_datetime(date_actualdata) - pd.to_datetime(date_datause)).days

predict_region_list = []

effect_ind = 0

for it in range(len(region)):

    region_it = region[it]

    if region_it not in no_infection_country:

        time_length_it = actual_date[effect_ind]

        sim_class_it = labels[effect_ind]

        predict_ip_it = sim_class_ip_array[sim_class_it,(len(actual_date[0])-date_length):]

        while len(predict_ip_it)< (date_length+31):

            predict_ip_it = np.append(predict_ip_it,predict_ip_it[len(predict_ip_it)-1])

        retion_df = data[data['PSCR'] == region_it]

        num_infected_it = retion_df[retion_df['Date'] == date_datause]['ConfirmedCases'].astype(float)

        predict_region_list_it = []

        ini_infected = num_infected_it.tolist()[0]

        for predict_day in range(len(predict_ip_it)):

            predict_region_list_it.append(ini_infected * (1+predict_ip_it[predict_day]))

            ini_infected = predict_region_list_it[predict_day]

        predict_region_list.extend(predict_region_list_it)

        effect_ind += 1

    else:

        predict_region_list.extend([0 for i in range(43)])



# %%

# Write output csv file

import csv

from itertools import zip_longest

list1 = [i+1 for i in range(len(predict_region_list))]

list2 = predict_region_list

list3 = [0 for i in range(len(predict_region_list))]

d = [list1, list2,list3]

export_data = zip_longest(*d, fillvalue = '')

with open('submission.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:

      wr = csv.writer(myfile)

      wr.writerow(("ForecastId", "ConfirmedCases", "Fatalities"))

      wr.writerows(export_data)

myfile.close()