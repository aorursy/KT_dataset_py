import subprocess
import pprint

sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

out_str = sp.communicate()
out_list = str(out_str[0]).split('\\n')

out_dict = {}

for item in out_list:
    print(item)
import subprocess
import pprint

sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

out_str = sp.communicate()
out_list = str(out_str[0]).split('\\n')

out_dict = {}

for item in out_list:
    print(item)