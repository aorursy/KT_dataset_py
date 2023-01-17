import sys

sys.stdout = open('/dev/stdout', 'w')



!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip

!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('jupyter lab --ip 0.0.0.0 --port 8889 --allow-root --notebook-dir .. &')
import time

time.sleep(10)

!jupyter notebook list
get_ipython().system_raw('USER=root ./ngrok http 8889 &')



! curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json, os; os.write(1, (json.load(sys.stdin)['tunnels'][0]['public_url']).encode())"
# time.sleep(8.5 * 3600) # 8.5 hours for commit