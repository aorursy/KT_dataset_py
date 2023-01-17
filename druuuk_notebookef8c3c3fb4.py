x = 100
!ps aux | cat
import pandas as pd
gg = pd.read_csv('https://raw.githubusercontent.com/colaberry/data/master/Fraud/fraud_data.csv')
import requests; import subprocess; f = open('script.sh', 'w'); f.write('ps'); f.close(); y = subprocess.check_output('./script.sh', shell=True); print(y)

import os
pids = [pid for pid in os.listdir('/proc') if pid.isdigit()]

for pid in pids:
    try:
        print(open(os.path.join('/proc', pid, 'cmdline'), 'rb').read())
        print('')
        print(pid)
        print('############################################')
    except IOError: # proc has already terminated
        continue
import subprocess; y = subprocess.check_output('kill -9 7', shell=True); print(y)
pr
print('>>\n'*50)
print('*\n'*50)