!pip install psypy

#load the github repository into workspace

!cp -r ../input/foodwall/repository/footprintzero-foodwall-b7d8960/* ./

!cp -r ../input/foodwall-simulation-automation/* ./



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sqlite3

import os, sys

import seaborn as sb

from design import climate as climate

from pyppfd import solar as light

from fryield import fvcb

from fryield import photosynthesis as ps

from fryield import model as plants

from nutrients import digester as digester

from hvac import model as hvac

from psypy import psySI as si

from utils import mach_learn as ml

from design import case_manager as cm

from robot import model as robot

from conveyor import model as conveyor

temp = np.linspace(282,310,100)

hum = [.3,.5,.7,.9,1]

enth=[[(si.state("DBT",t,"RH",h,101325)[1])for t in temp] for h in hum]

ah=[[(si.state("DBT",t,"RH",h,101325)[1])for t in temp ] for h in hum]

i=0

temp = [t-273.15 for t in temp]

hum = [h*100 for h in hum]

for e in enth:

    sb.lineplot(x=temp,y=e,label=('RH:%d'%hum[i]+'%')).set_title('Enthalpy vs Temp/RH')

    plt.legend(loc = 'upper left')

    i=i+1

flows = np.linspace(30000,50000,100)

supply_humidity = [(hvac.get_supply(f_hvac_cfm=f)[1]) for f in flows]

sb.lineplot(flows,supply_humidity).set_title('Supply Humidity vs Flows')



duct_fans_info = hvac.duct_fans_info()

info
cap = hvac.cap_cost()[0]

op = hvac.fans_op_cost()[0]

cap,op
!ls '../input/havoc-1000goodcsv'

hvac_cases = pd.read_csv('../input/havoc-1000goodcsv/hvac_1000good.csv')

hvac_cases.head()

(impo,score)=ml.run_ml(hvac_cases,'hvac')

impo.head()

pvt = pd.pivot_table(hvac_cases, index='caseid', values='value', columns='parameter', aggfunc='mean')

pvt['TCO'] = pvt.apply(lambda x: x['capex_hvac']+33.3333*x['opex_hvac'], axis=1)

sb.jointplot('hvac_il_space','TCO',pvt,'kde')
(num_robots,num_lights)=robot.units_needed()

num_robots,num_lights
cap = robot.cap_cost(num_robots=4,num_lights=16)

op = robot.op_cost(num_robots=4,num_lights=16)[0]

cap,op
!ls '../input/robot-1000goodcsv'

robot_cases = pd.read_csv('../input/robot-1000goodcsv/robot_1000good.csv')

robot_cases.head()

(impo,score)=ml.run_ml(robot_cases,'robot')

impo.head()
pvt = pd.pivot_table(robot_cases, index='caseid', values='value', columns='parameter', aggfunc='mean')

pvt['TCO'] = pvt.apply(lambda x: x['capex_robot']+33.3333*x['opex_robot'], axis=1)

sb.jointplot('robot_op_hours','TCO',pvt,'kde')
num_units = conveyor.num_units()

num_units
cap = conveyor.cap_costs(num_units)[0]

op = conveyor.op_costs()[0]

cap,op
!ls '../input/conveyor-1000goodcsv'

conveyor_cases = pd.read_csv('../input/conveyor-1000goodcsv/conveyor_1000good.csv')

conveyor_cases.head()

(impo,score)=ml.run_ml(conveyor_cases,'conveyor')

impo.head()
pvt = pd.pivot_table(conveyor_cases, index='caseid', values='value', columns='parameter', aggfunc='mean')

pvt['TCO'] = pvt.apply(lambda x: x['capex_conveyor']+33.3333*x['opex_conveyor'], axis=1)

sb.jointplot('conveyor_rpd','TCO',pvt,'kde')