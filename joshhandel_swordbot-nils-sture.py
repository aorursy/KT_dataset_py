#configure details for data processing run

#machine configurations

#----------------------

#step pulses sent by controller per 100mm of travel in the real world based on 1000 pulses per rev

pulses_per_100mm = 830 



#the servo motion controller records at 6400 pulses per rev for data recording

servo_pulses_per_100mm = pulses_per_100mm * 6.4 



#the experiment is zeroed to 600mm of travel to the surface of the ballistics gel

surface = 600



control_movement_end = 650



%config InlineBackend.figure_format = 'retina'
# get the calibration of newtons per measured step in the sensor

# here we average the 175 weight measurements that were taken and average them to 

# set a calibration

# to find force we will use measured_force/calibration_factor that we are calculating here

import numpy as np

import pandas as pd



calibration_data = pd.read_csv('../input/swordbot-materialtests/data/calibration.csv',header=None)



calibration_grouping = {}



calibration_array = calibration_data.to_numpy()

for row in calibration_array:

    if not (row[0] in calibration_grouping):

        calibration_grouping[row[0]] = []

    calibration_grouping[row[0]].append(row[1])

gravity = .009807

force_avgs = {'newtons':[],'measurement':[]}

for weight_group in calibration_grouping:

    force = weight_group * gravity

    force_avgs['newtons'].append(force)

    force_avgs['measurement'].append(round(np.mean(calibration_grouping[weight_group]),2))

zero_offset = force_avgs['measurement'][0]



df = pd.DataFrame(force_avgs)

display(df)



calibration_set = []

for i in range(len(force_avgs)):

    force_number = force_avgs['newtons'][i]

    measurement = force_avgs['measurement'][i]

    if not(force_number == 0.0):

        calibration_set.append((measurement-zero_offset)/force_number)

        

calibration_factor = round(np.mean(calibration_set),2)



display(str(calibration_factor) + " steps/newton")

#no-target-control configuration

#each run is made up of 3 files based on the run serial number

run_number = [371,372,373,374,383,384,385,386]



#array of all of the file names for the baseline runs

data_files ={"run_number":[],"servo_data_file":[],

             "controller_movement_log":[],

             "sensor_data_file":[],

             "servo_loc_key":[],

             "controller_time_key":[],

             "sensor_time_key":[]

            }

rollup_name = 'nils_sture'

for run in run_number:

    data_files['run_number'].append(run)

    #servo data is recorded by the servo. The servo records the actual encoded rotations that have been executed

    #and the error from that real world location with where it has been "commanded" to be by the controller

    data_files["servo_data_file"].append('../input/swordbot-materialtests/data/'+str(run)+'.xml')

    #movement log of each pulse sent to the servo, this data is used with the servo data to align the real world

    #movements with the logged load cell data.

    data_files["controller_movement_log"].append('../input/swordbot-materialtests/data/run-movement-log-'+str(run)+'.csv')

    #the load cell data is the actual force data logged by the sensor pack

    data_files["sensor_data_file"].append('../input/swordbot-materialtests/data/run-measurements-'+str(run)+'.csv')

    data_files["servo_loc_key"].append(-100000000)

    data_files["controller_time_key"].append(-100000000)

    data_files["sensor_time_key"].append(-100000000)



df = pd.DataFrame(data_files)

display(df)
#Sync 

import types

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [25, 5]

plt.rcParams.update({'font.size': 18})

from xml.etree import ElementTree as ET

servo_data = {}



def load_servo_log(file,index):

    servo_data = {

        #time the servo has reached the target (+/- backlash in the mechanical system)

        "time_at_surface":0, 

        #time the servo has received enough commands from the controller that it should "try" to be at the surface

        "time_at_commanded_surface":0, 

        #time at the end of the programmed motion (any movement after this is unprogrammed motion correction)

        "time_at_end": 0,

        #array of servo commanded and actual movements

        "log":{

            "time":[],     #time of datapoint

            "actual":[], #actual movement array

            "commanded":[] #commanded movement array

    }}

    #load the XML 

    tree = ET.parse(file)

    root = tree.getroot()

    #we need to correct the time by removing the "starting time" so our counter starts at 0

    time = 0

    #read the variable that holds the scale for movement error

    var0 = root.find('.//ScopeMain/DataAcq/Variable0')

    error_scale = float(var0.attrib["range"])

    #read the variable that holds the scale for actual movement

    var2 = root.find('.//ScopeMain/DataAcq/Variable2')

    servo_pulse_range = float(var2.attrib["range"])



    #flag to ignore log entries until we get some movement as the XML usually 

    #has a few milliseconds of non-movement data

    started_movement = False



    i = 0

    #loop through data

    for p in root.findall('.//ScopeMain/DataAcq/APnt'):



        if(not started_movement):

            if (float(p.attrib['val2']) > 0.0):

                # we finally look like we are starting to get some data.

                started_movement = True

                #this is the time in milliseconds (with microsecond percision) that 

                #we will consider the start of the log. 

                time = float(p.attrib['time'])



        if(started_movement):

            #adjust the row time so that is based on starting the reading at "0"

            #multiply to 1000 and round so that we go from milliseconds with 6 decimal places (more than microseconds)

            #to rounded microseconds

            rowtime = int((float(p.attrib["time"]) - time)*1000)



            val1 = float(p.attrib['val']) * error_scale # error from commanded to actual * scale

            val2 =  float(p.attrib['val2']) * servo_pulse_range # actual location of the servo * scale

            commanded = val1 + val2 #location that the controller has commanded the servo to be at

            

            #time of the log entry

            servo_data['log']['time'].append(rowtime)

            #actual movement converted from pulses to mm

            actual_movement = (val2 / servo_pulses_per_100mm) * 100

            servo_data['log']['actual'].append(actual_movement) 

            #commanded movement converted from pulses to mm

            commanded_movement = (commanded / servo_pulses_per_100mm) * 100

            servo_data['log']['commanded'].append(commanded_movement) 

            #if the servo has received enoough commands to reach the surface of the target

            #record that time

            if(servo_data['time_at_commanded_surface'] == 0 and commanded_movement >= surface):

                #because things move so fast we actually want the record before this (closer to 600 than this record)

                servo_data['time_at_commanded_surface'] = servo_data['log']['time'][i-1]

                data_files['servo_loc_key'][index] = servo_data['log']['commanded'][i-1]



            #if the servo has reached the surface of the target and we have not recorded a time yet

            #record that time

            if(servo_data['time_at_surface'] == 0 and actual_movement >= surface):

                #because things move so fast we actually want the record before this (closer to 600 than this record)

                servo_data['time_at_surface'] = servo_data['log']['time'][i-1]



            #if the servo has reached the end of the commanded movement and we have not recorded a time yet

            #record that time

            if(servo_data['time_at_end'] == 0 and actual_movement >= control_movement_end):

                #because things move so fast we actually want the record before this (closer to 600 than this record)

                servo_data['time_at_end'] = servo_data['log']['time'][i-1]

            i = i+1;

    return servo_data



servo_data['file_data'] = []

#parse out the data we need from each XML file (see method above)

for i in range(len(data_files["servo_data_file"])):

    servo_data['file_data'].append(load_servo_log(data_files['servo_data_file'][i],i))

    plt.subplot(2, 5, i+1)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.plot(servo_data['file_data'][i]['log']['time'],servo_data['file_data'][i]['log']["actual"])

    plt.plot(servo_data['file_data'][i]['log']['time'],servo_data['file_data'][i]['log']["commanded"]) 

    plt.title(rollup_name + ':Servo Data for Run ' + str(data_files['run_number'][i]))

df = pd.DataFrame(data_files)

display(df)
plt.title(rollup_name + ':Servo Data Stacked')

for i in range(len(data_files["servo_data_file"])):

    plt.plot(servo_data['file_data'][i]['log']['time'],servo_data['file_data'][i]['log']["actual"])

    plt.plot(servo_data['file_data'][i]['log']['time'],servo_data['file_data'][i]['log']["commanded"])

#recenter the data off of the time at commanded service, as this is where we will sync the 

#controller movement log

plt.title(rollup_name + ':Servo Data Time Aligned when travel reached 600mm')

for file_data in servo_data['file_data']:

    file_data['batch_offset'] = file_data['time_at_commanded_surface'] * -1

    #add a new array to store the offset time of each record.

    file_data['log']['offset_time'] = []

    for i in range(len(file_data['log']['time'])):

        file_data['log']['offset_time'].append(file_data['log']['time'][i] + file_data['batch_offset'])

        if(file_data['log']['time'][i] + file_data['batch_offset'] == 0):

            file_data['zero_index'] = i

    plt.plot(file_data['log']['offset_time'],file_data['log']["actual"])

    plt.plot(file_data['log']['offset_time'],file_data['log']["commanded"])

    
servo_data['ordered_data'] = []

#starting at zero append all the N positive data so it can be averaged (both time and measurements)

for file_data in servo_data['file_data']:

    for i in range(file_data['zero_index'],len(file_data['log']['time'])):

        new_index = i - file_data['zero_index']

        item = {}

        try:

            item = servo_data['ordered_data'][new_index]

        except IndexError:

            item = {}

            item['time_series'] = []

            item['actual_move_series'] = []

            item['commanded_move_series'] = []

            item['bucket'] = new_index

            servo_data['ordered_data'].append(item)

        item['time_series'].append(file_data['log']['offset_time'][i])

        item['actual_move_series'].append(file_data['log']['actual'][i])

        item['commanded_move_series'].append(file_data['log']['commanded'][i])
#starting at zero insert all the N negative data so it can be averaged (both time and measurements)

for file_data in servo_data['file_data']:

    processed_rows = -1;

    for i in range(file_data['zero_index']-1,0,-1):

        new_index = processed_rows + file_data['zero_index']

        item = {}

        item_index = -10000000

        for j in range(len(servo_data['ordered_data'])):

            if(servo_data['ordered_data'][j]['bucket'] == processed_rows):

                item = servo_data['ordered_data'][j]

                item_index = j

        if(item_index == -10000000):

            item = {}

            item['time_series'] = []

            item['actual_move_series'] = []

            item['commanded_move_series'] = []

            item['bucket'] = processed_rows

            servo_data['ordered_data'].insert(0,item)

        

        item['time_series'].append(file_data['log']['offset_time'][i])

        item['actual_move_series'].append(file_data['log']['actual'][i])

        item['commanded_move_series'].append(file_data['log']['commanded'][i])

        processed_rows = processed_rows - 1
#signal average the files and store the roled up new averaged data.

servo_data['average_signal'] = []



for data in servo_data['ordered_data']:

    data_row = [np.mean(data['time_series']),

                np.mean(data['actual_move_series']),

                np.mean(data['commanded_move_series'])]

    servo_data['average_signal'].append(data_row)



servo_average_df = pd.DataFrame(servo_data['average_signal'], 

                                         columns=["time","actual","commanded"])

try:

    os.remove('../working/rollups/rollup_'+rollup_name+'_servo_data.csv')

except:

    print("no file to delete")

    

servo_average_df.to_csv('../working/rollup_'+rollup_name+'_servo_data.csv')



sync_loc = 0;

for row in servo_data['average_signal']:

    if(row[0] == 0):

        sync_loc = row[2]



plt.title(rollup_name + ':Aligned Servo Data Averaged')

plt.plot(servo_average_df['time'],servo_average_df['actual'])

plt.plot(servo_average_df['time'],servo_average_df['commanded'])
#load the controller log captured by the controller (this shares the same internal clock with the load cell force data)

def load_controller_log(file,index):

    pulse_log = pd.read_csv(file,skiprows=5)

    servo_loc_key = data_files['servo_loc_key'][index]

    controller_data = {

        #time the controller sent the signal that should put the servo at the surface of the target

        'time_at_surface':0,

        #last command of the controller which is at the end of the controlled movement (650mm)

        'time_at_end':0,

        'log':{

            #microsecond the pulse was sent

            'time':[],

            #command normalized to mm

            'commanded':[] 

        },

    }

    for row in pulse_log.to_numpy():

        #time of the command

        time = int(row[0])

        command_loc = (int(row[1])/pulses_per_100mm) * 100

        controller_data['log']['time'].append(time)

        controller_data['log']['commanded'].append(command_loc) #command normalized to mm

        

        #if the controller has reached the surface movement command and we have not recorded a time yet

        #record that time

        if (command_loc >= servo_loc_key and controller_data['time_at_surface'] == 0):

            controller_data['time_at_surface'] = int(row[0])

            data_files["controller_time_key"][index] = controller_data['time_at_surface']



        #since this is always the last command just keep resetting the value till we get to the end

        #not elegant but no need to be here, its just an int

        controller_data['time_at_end'] = int(row[0])

    return controller_data



controller_data = {}

controller_data['file_data'] = []

#parse out the data we need from each CSV file (see method above)

for i in range(len(data_files['controller_movement_log'])):

    controller_data['file_data'].append(load_controller_log(data_files['controller_movement_log'][i],i))

    plt.subplot(2, 5, i+1)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.plot(controller_data['file_data'][i]['log']['time'],controller_data['file_data'][i]['log']["commanded"])

    plt.title(rollup_name + ':Controller Data for Run ' + str(data_files['run_number'][i]))

df = pd.DataFrame(data_files)

display(df)
plt.title(rollup_name + ':Controller Data Stacked')

for i in range(len(data_files['controller_movement_log'])):

    plt.plot(controller_data['file_data'][i]['log']['time'],controller_data['file_data'][i]['log']["commanded"])

    
#recenter the data off of the time at commanded servo zero, as this is where we will sync the 

#controller movement log

plt.title(rollup_name + ':Controller Data Time Aligned when travel reached 600mm')

for file_data in controller_data['file_data']:

    file_data['batch_offset'] = file_data['time_at_surface'] * -1

    #add a new array to store the offset time of each record.

    file_data['log']['offset_time'] = []

    for i in range(len(file_data['log']['time'])):

        file_data['log']['offset_time'].append(file_data['log']['time'][i] + file_data['batch_offset'])

        if(file_data['log']['time'][i] + file_data['batch_offset'] == 0):

            file_data['zero_index'] = i

    plt.plot(file_data['log']['offset_time'],file_data['log']["commanded"])
controller_data['ordered_data'] = []

#starting at zero append all the N positive data so it can be averaged (both time and measurements)

for file_data in controller_data['file_data']:

    for i in range(file_data['zero_index'],len(file_data['log']['time'])):

        new_index = i - file_data['zero_index']

        item = {}

        try:

            item = controller_data['ordered_data'][new_index]

        except IndexError:

            item = {}

            item['time_series'] = []

            item['controller_time_series'] = []

            item['commanded_move_series'] = []

            item['bucket'] = new_index

            controller_data['ordered_data'].append(item)

            

        item['controller_time_series'].append(file_data['log']['time'][i])

        item['commanded_move_series'].append(file_data['log']['commanded'][i])

        item['time_series'].append(file_data['log']['offset_time'][i])
#starting at zero insert all the N negative data so it can be averaged (both time and measurements)

for file_data in controller_data['file_data']:

    processed_rows = -1;

    for i in range(file_data['zero_index']-1,0,-1):

        new_index = processed_rows + file_data['zero_index']

        item = {}

        item_index = -10000000

        for j in range(len(controller_data['ordered_data'])):

            if(controller_data['ordered_data'][j]['bucket'] == processed_rows):

                item = controller_data['ordered_data'][j]

                item_index = j

        if(item_index == -10000000):

            item = {}

            item['time_series'] = []

            item['controller_time_series'] = []

            item['commanded_move_series'] = []

            item['bucket'] = processed_rows

            controller_data['ordered_data'].insert(0,item)



        

        item['controller_time_series'].append(file_data['log']['time'][i])

        item['commanded_move_series'].append(file_data['log']['commanded'][i])

        item['time_series'].append(file_data['log']['offset_time'][i])

        processed_rows = processed_rows - 1
import os

controller_data['average_signal'] = []



for data in controller_data['ordered_data']:

    data_row = [np.mean(data['time_series'])

                ,np.mean(data['controller_time_series'])

                ,np.mean(data['commanded_move_series'])]

    controller_data['average_signal'].append(data_row)



controller_average_df = pd.DataFrame(controller_data['average_signal'], 

                                         columns=["time","controller_time","commanded"])

try:

    os.remove('../working/rollup_'+rollup_name+'_command_data.csv')

except:

    print("no file to delete")

controller_average_df.to_csv('../working/rollup_'+rollup_name+'_command_data.csv')



controller_sync_loc = -10;

controller_time_sync = 0;

for row in controller_data['average_signal']:

    if(row[0] == 0):

        controller_sync_loc = row[2]

        controller_time_sync = row[1]

plt.title(rollup_name + ':Aligned Controller Data Averaged')

plt.plot(controller_average_df['time'],controller_average_df['commanded'])
#load the controller log captured by the controller (this shares the same internal clock with the load cell force data)

def load_sensor_log(file,index):

    data_log = pd.read_csv(file,skiprows=5)

    controller_time_key = data_files["controller_time_key"][index]

    sensor_data = {

        #time that the sensor timestamp is at or after the time calculated to be the surface of the material

        "time_at_surface":0,

        "log":{

            "time":[],

            "force":[]

        }

    }



    for row in data_log.to_numpy():

        #time of the command

        time = int(row[0])

        force = int(row[1])/calibration_factor #force/calibration = newtons

        sensor_data['log']['time'].append(time)

        sensor_data['log']['force'].append(force)

        

        #if the controller has reached the surface movement command and we have not recorded a time yet

        #record that time

        if (time >= controller_time_key and sensor_data['time_at_surface'] == 0):

            sensor_data['time_at_surface'] = int(row[0])

            data_files["sensor_time_key"][index] = sensor_data['time_at_surface']

    return sensor_data



sensor_data = {}

sensor_data['file_data'] = []

#parse out the data we need from each CSV file (see method above)

for i in range(len(data_files['sensor_data_file'])):

    sensor_data['file_data'].append(load_sensor_log(data_files['sensor_data_file'][i],i))

    plt.subplot(2, 5, i+1)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.plot(sensor_data['file_data'][i]['log']['time'],sensor_data['file_data'][i]['log']["force"])

    plt.title(rollup_name + ':Sensor' + str(data_files['run_number'][i]))

df = pd.DataFrame(data_files)

display(df)
plt.title(rollup_name + ':Sensor Data Stacked')

for i in range(len(data_files['sensor_data_file'])):

    plt.plot(sensor_data['file_data'][i]['log']['time'],sensor_data['file_data'][i]['log']["force"])
#recenter the data off of the time at controller log, as this is where we will sync the 

plt.title(rollup_name + ':Sensor Data Time Aligned when travel reached 600mm')

for file_data in sensor_data['file_data']:

    file_data['batch_offset'] = file_data['time_at_surface'] * -1

    #add a new array to store the offset time of each record.

    file_data['log']['offset_time']=[]

    for i in range(len(file_data['log']['time'])):

        file_data['log']['offset_time'].append(file_data['log']['time'][i] + file_data['batch_offset'])

        if(file_data['log']['time'][i] + file_data['batch_offset'] == 0):

            file_data['zero_index'] = i

    plt.plot(file_data['log']['offset_time'],file_data['log']["force"])
sensor_data['ordered_data'] = []

#starting at zero append all the N positive data so it can be averaged (both time and measurements)

for file_data in sensor_data['file_data']:

    for i in range(file_data['zero_index'],len(file_data['log']['time'])):

        new_index = i - file_data['zero_index']

        item = {}

        try:

            item = sensor_data['ordered_data'][new_index]

        except IndexError:

            item = {}

            item['time_series'] = []

            item['force'] = []

            item['original_time_series'] = []

            item['bucket'] = new_index

            sensor_data['ordered_data'].append(item)

            

        item['original_time_series'].append(file_data['log']['time'][i])

        item['force'].append(file_data['log']['force'][i])

        item['time_series'].append(file_data['log']['offset_time'][i])
#starting at zero insert all the N negative data so it can be averaged (both time and measurements)

for file_data in sensor_data['file_data']:

    processed_rows = -1;

    for i in range(file_data['zero_index']-1,0,-1):

        new_index = processed_rows + file_data['zero_index']

        item = {}

        item_index = -10000000

        for j in range(len(sensor_data['ordered_data'])):

            if(sensor_data['ordered_data'][j]['bucket'] == processed_rows):

                item = sensor_data['ordered_data'][j]

                item_index = j

        if(item_index == -10000000):

            item = {}

            item['time_series'] = []

            item['force'] = []

            item['original_time_series'] = []

            item['bucket'] = processed_rows

            sensor_data['ordered_data'].insert(0,item)

        

        item['original_time_series'].append(file_data['log']['time'][i])

        item['force'].append(file_data['log']['force'][i])

        item['time_series'].append(file_data['log']['offset_time'][i])

        processed_rows = processed_rows - 1
sensor_data['average_signal'] = []



for data in sensor_data['ordered_data']:

    data_row = [np.mean(data['time_series']),np.mean(data['original_time_series']),np.mean(data['force'])]

    sensor_data['average_signal'].append(data_row)



sensor_average_df = pd.DataFrame(sensor_data['average_signal'], 

                                         columns=["time","original_time","force"])

try:

    os.remove('../working/rollup_'+rollup_name+'_sensor_data.csv')

except:

    print("no file to delete")

sensor_average_df.to_csv('../working/rollup_'+rollup_name+'_sensor_data.csv')

plt.title(rollup_name + ':Aligned Sensor Data Averaged')

plt.plot(sensor_average_df['time'],sensor_average_df['force'])

sensor_average_df
#load up the baseline

from scipy.signal import savgol_filter

baseline_df = pd.read_csv('../input/swordbot-baseline/rollup_baseline_sensor_data.csv')

savgol_filtered_baseline_data = savgol_filter(baseline_df['force'], 51, 2)
from scipy.signal import savgol_filter



#get the time when the servo crossed the end of control

servo_time_at_end = 0

for row in servo_data['average_signal']:

    if(row[1] >= 650):

        servo_time_at_end = row[0]

        break



fig, ax1 = plt.subplots()

#first chart is raw load cell data

ax1.plot(sensor_average_df["time"],sensor_average_df["force"],color="gray", label="Signal Averaged(newtons)")

ax1.plot(baseline_df["time"],savgol_filtered_baseline_data)

ax1.set_xlabel('time (microseconds)')

# Make the y-axis label, ticks and tick labels match the line color.

ax1.set_ylabel('force(N)', color='b')



savgol_filtered_data = savgol_filter(sensor_average_df['force'], 51, 2)

ax1.plot(sensor_average_df["time"],savgol_filtered_data,color="red", label="Savitzky–Golay(newtons)")

display(len(savgol_filtered_data))

#add some verticle at target surface & movement end times

ax1.axvspan(-6000, servo_time_at_end+6000, alpha=0.1, color='green', label='investigation zone')

ax1.axvspan(0, servo_time_at_end, alpha=0.25, color='green', label='time of interaction zone')

ax1.axvspan(servo_time_at_end, servo_time_at_end+108000, alpha=0.10, color='purple', label='servo deceleration')





#second chart is of the movement data from servo and controller

ax2 = ax1.twinx()

ax2.plot(servo_average_df["time"],servo_average_df["actual"],color='blue', label="Servo Travel(mm)")

ax2.set_ylabel('mm', color='blue')



#ax2.plot(servo_average_df["time"],servo_average_df["commanded_loc"],color='green')

#ax2.plot(controller_average_df["time"],controller_average_df["commanded_loc"],color='b')



#add some horizontal lines at target surface & controlled movement end distances 

#a perfect movement will cut the rectangle created by these lines

#and the lines above into two right triangles

ax2.axhspan(600,650, alpha=0.1, color="blue", label="travel of interaction zone)")





fig.tight_layout()

plt.title(rollup_name + ':Aligned Sensor Data with Savitzky–Golay filter and meta-data overlays')

ax2.legend(loc="upper right")

leg = ax1.legend(loc='upper left')

leg.remove()

ax2.add_artist(leg)

fig.patch.set_facecolor('white')

plt.plot()

#get baseline force median using filtered data

baseline_force_set=[]

for i in range(len(baseline_df["time"])):

    row = baseline_df["time"][i]

    if(row > 0 and row < servo_time_at_end):

        baseline_force_set.append(savgol_filtered_baseline_data[i])



#median force here because this is a range of noise where "nothing" is going on

baseline_force = np.average(baseline_force_set) 

baseline_variance = np.var(baseline_force_set)

display(baseline_force)

display(baseline_variance)

display(np.std(baseline_force_set))
#get peak force in the investigation zone (interaction zone +/- 6 milliseconds)

peak_force = -1000000

peak_force_index = -1000000

for i in range(len(sensor_data['average_signal'])):

    row = sensor_data['average_signal'][i]

    if(row[0] > 0 -6000 and row[0] <= servo_time_at_end+6000):

        if(savgol_filtered_data[i] > peak_force):

            peak_force = savgol_filtered_data[i] 

            peak_force_index = i



peak_force_time = sensor_average_df["time"][peak_force_index]

net_force = peak_force - baseline_force

net_force
#because the peak force is outside of our interaction zone, and there is a clear downward movement in the baseline

#net force is actually based on that the correct time mached datapoint in the baseline

baseline_time_found = False

for i in range(len(baseline_df["time"])):

    row = baseline_df["time"][i]

    if(row > peak_force_time and baseline_time_found == False):

        #sence we are doing greater than we go back 1 measurement as that is usually closer in alignment

        baseline_force = savgol_filtered_baseline_data[i -1] 

        baseline_time_found = True

        break

net_force = peak_force - baseline_force

net_force
peak_force_time - servo_time_at_end
fig, ax1 = plt.subplots()

#first chart is raw load cell data

ax1.plot(sensor_average_df["time"],sensor_average_df["force"],color="gray", alpha=.25, label="Signal Averaged(newtons)")

ax1.plot(baseline_df["time"],savgol_filtered_baseline_data)

ax1.set_xlabel('time (microseconds)')

# Make the y-axis label, ticks and tick labels match the line color.

ax1.set_ylabel('force(N)', color='b')



savgol_filtered_data = savgol_filter(sensor_average_df['force'], 51, 2)

ax1.plot(sensor_average_df["time"],savgol_filtered_data,color="red", label="Savitzky–Golay(newtons)")



#add some verticle at target surface & movement end times

ax1.axvspan(-6000, servo_time_at_end+6000, alpha=0.1, color='green', label='investigation zone')

ax1.axvspan(0, servo_time_at_end, alpha=0.25, color='green', label='time of interaction zone')

ax1.axvspan(servo_time_at_end, servo_time_at_end+108000, alpha=0.10, color='purple', label='servo deceleration')

ax1.axhline(baseline_force,linestyle=':', label="average measurement in interaction zone")

ax1.axhline(peak_force, linestyle="-.", label="peak force measured")

ax1.plot([peak_force_time,peak_force_time],[baseline_force,peak_force], label="net force")





bbox = dict(boxstyle="round", color="white")

ax1.annotate("Net force:" + str(round(net_force,2)) + "n"

             ,xy=(peak_force_time-1400,(net_force/2)+baseline_force)

             ,bbox=bbox, fontsize=15)

fig.tight_layout()

plt.title(rollup_name + ':Closeup view of the investigation zone with peak force highlighted and net force calculated')

leg = ax1.legend(loc='upper left')

fig.patch.set_facecolor('white')

plt.xlim(-7500,servo_time_at_end + 7500)

plt.ylim(-10,net_force+5)

plt.plot()