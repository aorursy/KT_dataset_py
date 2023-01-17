import pickle

import zlib

import base64 as b64

import numpy as np



def serializeAndCompress(value, verbose=True):

  serializedValue = pickle.dumps(value)

  if verbose:

    print('Lenght of serialized object:', len(serializedValue))

  c_data =  zlib.compress(serializedValue, 9)

  if verbose:

    print('Lenght of compressed and serialized object:', len(c_data))

  return b64.b64encode(c_data)



def decompressAndDeserialize(compresseData):

  d_data_byte = b64.b64decode(compresseData)

  data_byte = zlib.decompress(d_data_byte)

  value = pickle.loads(data_byte)

  return value
m = 12

n = 12

obs_space_n = m*n

action_space_n = 2

q_table = np.zeros([obs_space_n, action_space_n])

for n_sim in range(10000):

    # Train your agent

    # ...

    # Store a fake Qtable value

    q_table[np.random.randint(obs_space_n), np.random.randint(action_space_n)] = np.random.random()

serialized_q_table = serializeAndCompress(q_table)
print(serialized_q_table)
deserialized_q_table = decompressAndDeserialize(serialized_q_table)

print(deserialized_q_table[:10,:])