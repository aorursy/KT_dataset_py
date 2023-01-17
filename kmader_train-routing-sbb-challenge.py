!git clone https://github.com/crowdAI/train-schedule-optimisation-challenge-starter-kit
import os, sys, shutil
sk_dir = os.path.join('train-schedule-optimisation-challenge-starter-kit')

util_dir = os.path.join(sk_dir, 'utils')
sys.path.append(util_dir)
# we want the csv files in the same directory so we can validate the solution
!cp {util_dir}/*.csv . 
!cp {util_dir}/*.py . 
data_dir = os.path.join(sk_dir, 'problem_instances')
!cp -r {data_dir} .

# import a few of the tools
from route_graph import generate_route_graphs, save_graph
from validate_solution import do_loesung_validation
from glob import glob
import json
prob_files = glob(os.path.join(data_dir, '*.json'))
sample_files = glob(os.path.join(data_dir, '*', '*.json'))
test_file = prob_files[0]
sample_out_file = glob(os.path.join(data_dir, '*', 'sample*.json'))[0]
def read_problem(in_path):
    with open(in_path) as f:
        print('loading', in_path)
        out_vals = json.load(f)
        for k, v in out_vals.items():
            print('\t',k, v 
                  if k in ['hash', 'label']  else 'length: {}'.format(len(v)))
        return out_vals
print(len(prob_files), 'problems to solve')
print(len(sample_files), 'solutions')
test_json = read_problem(test_file)
from IPython.display import clear_output
route_graphs = generate_route_graphs(test_json)
clear_output() # very noisy function
print(len(route_graphs), 'routes')
save_graph({k:v for _, (k, v) in zip(range(4), route_graphs.items())})
test_json['service_intentions'][0]
!rm -rf {sk_dir}