#User can edit this value, to define time span:

time_span=300
import requests  # pip3 install requests

import json

request_data = {

    'method': 'server_info',

    'params': [

        {}

    ]

}

import time

from datetime import datetime, timedelta



validation_metrics_ls=[]

last_retrieved_seq = 0

minimum_validation_time = timedelta(seconds=5)

maximum_validation_time = timedelta(seconds=-5)

average_validation_time = minimum_validation_time



i = 0

time_end = time.time()+time_span

polling_interval = 1.5

print("Starting to collect data from RippleD at: ",datetime.now()," please wait ", time_span, " seconds...")



while(time.time() < time_end):

    response = requests.post('http://s1.ripple.com:51234/', headers={'Content-type': 'application/json'}, data=json.dumps(request_data))

    response.close()



    if response.status_code == 200:

        response_data = json.loads(response.text)

    else:

        print("Query operation to Ripple daemon failed. Check your parameters. To verify server's current status you can use [ping] method.")



    validated_ledger_seq = str(

            response_data['result']['info']['validated_ledger']['seq'])



    if validated_ledger_seq == last_retrieved_seq:

        polling_interval += 0.5

        print("Same ledger identified [",validated_ledger_seq,"]. Running too fast. Slow down!!! New polling interval is: [",polling_interval,"]")

    else:

        if i == 0:

            last_retrieved_time = datetime.strptime(

                response_data['result']['info']['time'], "%Y-%b-%d %H:%M:%S.%f %Z")

            latest_retrieved_time = last_retrieved_time

            time_difference = latest_retrieved_time-last_retrieved_time

        else:

            # Updating polling interval:

            ledger_difference = int(

                validated_ledger_seq)-int(last_retrieved_seq)

            if ledger_difference == 1:

                #No changes required. Putting this condition only for better readability.

                pass

            elif ledger_difference == 0:

                #Polling_interval already updated in above if check. Putting this condition only for better readability.

                pass

            elif ledger_difference > 1:

                polling_interval /= ledger_difference



            latest_retrieved_time = datetime.strptime(

                response_data['result']['info']['time'], "%Y-%b-%d %H:%M:%S.%f %Z")

            time_difference = latest_retrieved_time-last_retrieved_time

            last_retrieved_time = latest_retrieved_time



            if time_difference < minimum_validation_time:

                minimum_validation_time = time_difference



            if time_difference > maximum_validation_time:

                maximum_validation_time = time_difference



            last_retrieved_seq = validated_ledger_seq

            validation_metrics_ls.append('\n'+response_data['result']['info']['time'] + ',' + str(time_difference.seconds) + '.'+str(time_difference.microseconds) + ',' + validated_ledger_seq)

    i+=1

    time.sleep(polling_interval)

print("Data collection completed at: ",datetime.now())

print(i," requests made to RippleD server, during ", time_span," seconds.")



average_validation_time = (minimum_validation_time+maximum_validation_time)/2

print("Fastest ledger closed in: [",minimum_validation_time,"] seconds.")
print("Minimum time it took for a new ledger to be validated during last ",time_span," seconds: ",minimum_validation_time)

print("Maximum time it took for a new ledger to be validated during last ",time_span," seconds: ",maximum_validation_time)

print("Average time it took for a new ledger to be validated during last ",time_span," seconds: ",average_validation_time)
validation_metrics_csv = "rippled_response_"+str(time.time())+".csv"

validation_metrics_csv_obj = open(validation_metrics_csv, "w")

validation_metrics_csv_obj.write('time,time_difference,validated_ledger_seq')



for metric in validation_metrics_ls:

    validation_metrics_csv_obj.write(metric)



validation_metrics_csv_obj.close()

print("Data written to: "+validation_metrics_csv)
from matplotlib import pyplot as plt  # pip3 install matplotlib

import numpy as np

import pandas as pd # pip3 install pandas



validation_metrics_df = pd.read_csv(validation_metrics_csv) 

validation_metrics_df = validation_metrics_df[['time_difference', 'validated_ledger_seq']].copy()

validation_metrics_df=validation_metrics_df.astype(np.float)

validation_metrics_df.sort_values('time_difference',inplace=True)

validation_metrics_df.head()
axis = plt.gca()

figure = plt.gcf()



#validation_metrics_df.plot(kind='line',x='time_difference',y='validated_ledger_seq',ax=axis)

figure.set_size_inches(18.5, 10.5)

validation_metrics_df.plot(kind='line',x='time_difference',y='validated_ledger_seq', color='red', ax=axis)



plt.xlabel('Time taken to close ledger\n[s.ms]', fontsize=14)

plt.ylabel('Ledger sequence number', fontsize=12)

plt.yticks(validation_metrics_df['validated_ledger_seq'])

plt.setp(axis.get_xticklabels(), rotation=90)



plt.show()
validation_metrics_df.sort_values('validated_ledger_seq',inplace=True)

axis = plt.gca()

figure = plt.gcf()



figure.set_size_inches(18.5, 10.5)

validation_metrics_df.plot(kind='line',x='time_difference',y='validated_ledger_seq', color='red', ax=axis)



plt.xlabel('Time taken to close ledger\n[s.ms]', fontsize=14)

plt.ylabel('Ledger sequence number', fontsize=12)

plt.yticks(validation_metrics_df['validated_ledger_seq'])

plt.setp(axis.get_xticklabels(), rotation=90)



plt.show()
import seaborn as sns

plt.figure(figsize=(15,5))

axis=sns.violinplot(y='time_difference', data=validation_metrics_df,orient="h")

axis.set(xlabel='Time Difference', ylabel='Frequency of Ledger closing')
plt.figure(figsize=(15,5))

axis=sns.boxplot(y='time_difference', data=validation_metrics_df,orient="h")

axis.set(xlabel='Time Difference', ylabel='Frequency of Ledger closing')