import json



id_map = dict()

inv_id_map = dict()

id_count = 0
print("Reading json")

with open('dataset/starwars-full-interactions.json', 'r') as f:

    json_data = json.load(f)



print("----- Creating Interaction Edgelist -----")

print("Creating name map")

json_node = json_data["nodes"]

for character in json_node:

    id_map[character["name"]] = id_count

    inv_id_map[id_count] = character["name"]

    id_count += 1



print("Creating edges with weight")

json_link = json_data["links"]

line = open('interaction_link_with_weight.edgelist', 'w')

for character in json_link:

    start = character["source"]

    end = character["target"]

    weight = float(character["value"])

    line.write('{} {} {}\n'.format(start, end, weight))

line.close()

print('Interaction Edgelist is created!')
print("----- Creating Mentions Edgelist -----")



id_map_m_unordered = dict()

inv_id_map_m_unordered = dict()

id_map_m = dict()

inv_id_map_m = dict()

id_count_m = 0



with open('dataset/starwars-full-mentions.json', 'r') as f:

    json_data_m = json.load(f)

print("Creating name map")

json_node_m = json_data_m["nodes"]

for character in json_node_m:

    if character["name"] in id_map:

        id_map_m[character["name"]] = id_map[character["name"]]

        inv_id_map_m[id_map[character["name"]]] = character["name"]



        id_map_m_unordered[character["name"]] = id_count_m

        inv_id_map_m_unordered[id_count_m] = character["name"]

        id_count_m += 1



print("Creating edges with weight")

json_link_m = json_data_m["links"]

line = open('mentions_link_with_weight.edgelist', 'w')



for character in json_link_m:

    if character["source"] < len(id_map) and character["target"] < len(id_map):

        start = id_map[inv_id_map_m_unordered[character["source"]]]

        end = id_map[inv_id_map_m_unordered[character["target"]]]

        weight = float(character["value"])

        line.write('{} {} {}\n'.format(start, end, weight))

line.close()



print('Mentions Edgelist is created!')
# Create id_map.json, inv_id_map.json

json_file = json.dumps(id_map)

f = open("id_map.json","w")

f.write(json_file)

f.close()

print('ID Map is Created!')



json_file = json.dumps(inv_id_map)

f = open("inv_id_map.json","w")

f.write(json_file)

f.close()

print('Inv Map is Created!')
import json

import numpy as np
feat = np.loadtxt('node2vec/output/aggregated_feat.txt')

feat_id = feat[:,0].astype(int)
json_file = open('id_map.json','r')    

for line in json_file:

    id_map = json.loads(line)

    

json_file = open('inv_id_map.json','r')

for line in json_file:

    inv_id_map = json.loads(line)
def feat_distance(character_1, character_2):

    idx_1 = np.nonzero(feat_id==id_map[character_1])[0][0]

    idx_2 = np.nonzero(feat_id==id_map[character_2])[0][0]

    

    feature_1 = np.copy(feat[idx_1, 1:])

    feature_2 = np.copy(feat[idx_2, 1:])

    

    return np.linalg.norm(feature_1 - feature_2)
def k_nn_characters(character, k):

    idx = np.nonzero(feat_id==id_map[character])[0][0]

    feat_vec = np.copy(feat[idx, 1:])



    dist = np.power(feat[:, 1:] - feat_vec, 2)

    dist = np.sum(dist, axis = 1)

    

    sort_list = np.argsort(dist)

    friend_list = sort_list[1:k+1]

    for i in friend_list:

        print(inv_id_map[str(i)])
print('5 most closest friends of OBI-WAN!')

k_nn_characters('OBI-WAN', 5)
print('5 most closest friends of DARTH VADER')

k_nn_characters('DARTH VADER', 5)