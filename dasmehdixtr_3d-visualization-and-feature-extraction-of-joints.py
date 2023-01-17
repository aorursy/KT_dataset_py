!pip install transforms3d
import numpy as np

import matplotlib.pyplot as plt

from transforms3d.euler import euler2mat

from mpl_toolkits.mplot3d import Axes3D





class Joint:

  def __init__(self, name, direction, length, axis, dof, limits):

    """

    Definition of basic joint. The joint also contains the information of the

    bone between it's parent joint and itself. Refer

    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)

    for detailed description for asf files.



    Parameter

    ---------

    name: Name of the joint defined in the asf file. There should always be one

    root joint. String.



    direction: Default direction of the joint(bone). The motions are all defined

    based on this default pose.



    length: Length of the bone.



    axis: Axis of rotation for the bone.



    dof: Degree of freedom. Specifies the number of motion channels and in what

    order they appear in the AMC file.



    limits: Limits on each of the channels in the dof specification



    """

    self.name = name

    self.direction = np.reshape(direction, [3, 1])

    self.length = length

    axis = np.deg2rad(axis)

    self.C = euler2mat(*axis)

    self.Cinv = np.linalg.inv(self.C)

    self.limits = np.zeros([3, 2])

    for lm, nm in zip(limits, dof):

      if nm == 'rx':

        self.limits[0] = lm

      elif nm == 'ry':

        self.limits[1] = lm

      else:

        self.limits[2] = lm

    self.parent = None

    self.children = []

    self.coordinate = None

    self.matrix = None



  def set_motion(self, motion):

    if self.name == 'root':

      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])

      rotation = np.deg2rad(motion['root'][3:])

      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)

    else:

      idx = 0

      rotation = np.zeros(3)

      for axis, lm in enumerate(self.limits):

        if not np.array_equal(lm, np.zeros(2)):

          rotation[axis] = motion[self.name][idx]

          idx += 1

      rotation = np.deg2rad(rotation)

      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)

      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)

    for child in self.children:

      child.set_motion(motion)



  def draw(self):

    joints = self.to_dict()

    fig = plt.figure()

    ax = Axes3D(fig)



    ax.set_xlim3d(-50, 10)

    ax.set_ylim3d(-20, 40)

    ax.set_zlim3d(-20, 40)

    ax.set_xlabel('X Label')

    ax.set_ylabel('Y Label')

    ax.set_zlabel('Z Label')



    xs, ys, zs = [], [], []

    for joint in joints.values():

      xs.append(joint.coordinate[0, 0])

      ys.append(joint.coordinate[1, 0])

      zs.append(joint.coordinate[2, 0])

    plt.plot(zs, xs, ys, 'b.')

    #print('xs = {}\n'.format(xs))

    for joint in joints.values():

      child = joint

      if child.parent is not None:

        parent = child.parent

        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]

        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]

        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]

        plt.plot(zs, xs, ys, 'r')

    plt.show()

    

    

  def to_dict(self):

    ret = {self.name: self}

    for child in self.children:

      ret.update(child.to_dict())

    return ret



  def pretty_print(self):

    print('===================================')

    print('joint: %s' % self.name)

    print('direction:')

    print(self.direction)

    print('limits:', self.limits)

    print('parent:', self.parent)

    print('children:', self.children)





def read_line(stream, idx):

  if idx >= len(stream):

    return None, idx

  line = stream[idx].strip().split()

  idx += 1

  return line, idx





def parse_asf(file_path):

  '''read joint data only'''

  with open(file_path) as f:

    content = f.read().splitlines()



  for idx, line in enumerate(content):

    # meta infomation is ignored

    if line == ':bonedata':

      content = content[idx+1:]

      break



  # read joints

  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}

  idx = 0

  while True:

    # the order of each section is hard-coded



    line, idx = read_line(content, idx)



    if line[0] == ':hierarchy':

      break



    assert line[0] == 'begin'



    line, idx = read_line(content, idx)

    assert line[0] == 'id'



    line, idx = read_line(content, idx)

    assert line[0] == 'name'

    name = line[1]



    line, idx = read_line(content, idx)

    assert line[0] == 'direction'

    direction = np.array([float(axis) for axis in line[1:]])



    # skip length

    line, idx = read_line(content, idx)

    assert line[0] == 'length'

    length = float(line[1])



    line, idx = read_line(content, idx)

    assert line[0] == 'axis'

    assert line[4] == 'XYZ'



    axis = np.array([float(axis) for axis in line[1:-1]])



    dof = []

    limits = []



    line, idx = read_line(content, idx)

    if line[0] == 'dof':

      dof = line[1:]

      for i in range(len(dof)):

        line, idx = read_line(content, idx)

        if i == 0:

          assert line[0] == 'limits'

          line = line[1:]

        assert len(line) == 2

        mini = float(line[0][1:])

        maxi = float(line[1][:-1])

        limits.append((mini, maxi))



      line, idx = read_line(content, idx)



    assert line[0] == 'end'

    joints[name] = Joint(

      name,

      direction,

      length,

      axis,

      dof,

      limits

    )



  # read hierarchy

  assert line[0] == ':hierarchy'



  line, idx = read_line(content, idx)



  assert line[0] == 'begin'



  while True:

    line, idx = read_line(content, idx)

    if line[0] == 'end':

      break

    assert len(line) >= 2

    for joint_name in line[1:]:

      joints[line[0]].children.append(joints[joint_name])

    for nm in line[1:]:

      joints[nm].parent = joints[line[0]]



  return joints





def parse_amc(file_path):

  with open(file_path) as f:

    content = f.read().splitlines()



  for idx, line in enumerate(content):

    if line == ':DEGREES':

      content = content[idx+1:]

      break



  frames = []

  idx = 0

  line, idx = read_line(content, idx)

  assert line[0].isnumeric(), line

  EOF = False

  while not EOF:

    joint_degree = {}

    while True:

      line, idx = read_line(content, idx)

      if line is None:

        EOF = True

        break

      if line[0].isnumeric():

        break

      joint_degree[line[0]] = [float(deg) for deg in line[1:]]

    frames.append(joint_degree)

  return frames





def test_all():

  import os

  lv0 = './data'

  lv1s = os.listdir(lv0)

  for lv1 in lv1s:

    lv2s = os.listdir('/'.join([lv0, lv1]))

    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)

    print('parsing %s' % asf_path)

    joints = parse_asf(asf_path)

    motions = parse_amc('./nopose.amc')

    joints['root'].set_motion(motions[0])

    joints['root'].draw()



    # for lv2 in lv2s:

    #   if lv2.split('.')[-1] != 'amc':

    #     continue

    #   amc_path = '%s/%s/%s' % (lv0, lv1, lv2)

    #   print('parsing amc %s' % amc_path)

    #   motions = parse_amc(amc_path)

    #   for idx, motion in enumerate(motions):

    #     print('setting motion %d' % idx)

    #     joints['root'].set_motion(motion)





if __name__ == '__main__':

  #test_all()

  asf_path = '/kaggle/input/cmu-motion-capture-walking-database/35.asf'

  amc_path = '/kaggle/input/cmu-motion-capture-walking-database/walk/07_11.amc'

  joints = parse_asf(asf_path)

  print(joints.values())

  motions = parse_amc(amc_path)

  

  frame_idx = 100

  joints['root'].set_motion(motions[frame_idx])

  joints['root'].draw() 

  
def combineMultipleDatas(data_names):

   datas = data_names[0]

   x = 0

   for data in data_names:

       if x == 0:

           result = datas.append(data,ignore_index=True)

       else:

           result = result.append(data,ignore_index=True)

       x = x+ 1

   return result

def dataPreparer(df,df2,df3,liste,liste2,liste3):

    a = 1

    b = 0

    

    for i in liste: 

        df.iloc[b,(a%31-1)] = i

        if a % 31 ==0:

            b = b+1

        #print(b,a)

        a = a+1

    a = 1

    b = 0

    for i in liste2:

        df2.iloc[b,(a%31-1)] = i

        if a % 31 ==0:

            b = b+1

        #print(b,a)

        a = a+1

    a = 1

    b = 0

    for i in liste3:

        df3.iloc[b,(a%31-1)] = i

        if a % 31 ==0:

            b = b+1

        #print(b,a)

        a = a+1



    data = pd.concat([df,df2],axis=1,ignore_index=True)

    data = pd.concat([data,df3],axis=1,ignore_index=True)

    return data



import pandas as pd

df = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,34))

df2 = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,34))

df3 =pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,34))

liste= []

liste2 = []

liste3 = []

import os
if __name__ == '__main__':

    asf_path = '/kaggle/input/cmu-motion-capture-walking-database/35.asf'

    path ='/kaggle/input/cmu-motion-capture-walking-database/walk_2/'

    for entry in sorted(os.listdir(path)):

      if os.path.isfile(os.path.join(path, entry)):

          joints = parse_asf(asf_path)

          motions = parse_amc(path+entry)

          frame_idx= 0

          joints['root'].set_motion(motions[frame_idx])

          #joints['root'].draw()

          

          for j in joints.items():

              for k in j[1].coordinate.T:

                  liste.append(k[0])

                  liste2.append(k[1])

                  liste3.append(k[2])

    data = dataPreparer(df,df2,df3,liste,liste2,liste3)
if __name__ == '__main__':

    asf_path = '/kaggle/input/cmu-motion-capture-walking-database/07.asf'

    path ='/kaggle/input/cmu-motion-capture-walking-database/walk/'

    liste4 = []

    liste5 = []

    liste6 = []

    for entry in sorted(os.listdir(path)):

      if os.path.isfile(os.path.join(path, entry)):

          joints = parse_asf(asf_path)

          motions = parse_amc(path+entry)

          frame_idx= 0

          joints['root'].set_motion(motions[frame_idx])

          #joints['root'].draw()

          for js in joints.items():

              for ks in js[1].coordinate.T:

                  liste4.append(ks[0])

                  liste5.append(ks[1])

                  liste6.append(ks[2])

    df4 = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,12))

    df5 = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,12))

    df6 =pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,12))

    data2 = dataPreparer(df4,df5,df6,liste4,liste5,liste6)
if __name__ == '__main__':

    asf_path = '/kaggle/input/cmu-motion-capture-walking-database/39.asf'

    path ='/kaggle/input/cmu-motion-capture-walking-database/walk_3/'

    liste7 = []

    liste8 = []

    liste9 = []

    for entry in sorted(os.listdir(path)):

      if os.path.isfile(os.path.join(path, entry)):

          joints = parse_asf(asf_path)

          motions = parse_amc(path+entry)

          frame_idx= 0

          joints['root'].set_motion(motions[frame_idx])

          #joints['root'].draw()

          for js in joints.items():

              for ks in js[1].coordinate.T:

                  liste7.append(ks[0])

                  liste8.append(ks[1])

                  liste9.append(ks[2])

    df7 = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,14))

    df8 = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,14))

    df9 = pd.DataFrame(columns = np.arange(1,32),index= np.arange(0,14))

    data4 = dataPreparer(df7,df8,df9,liste7,liste8,liste9)
data3 = pd.concat([data,data2,data4],ignore_index=True)
x = data3

from sklearn.decomposition import PCA



pca = PCA(n_components=31) #whitten = normalize

pca.fit(x)



x_pca = pca.transform(x)



print("variance ratio: ",pca.explained_variance_ratio_)



print("sum: ",sum(pca.explained_variance_ratio_))
data3["p1"] = x_pca[:,0]

data3["p2"] = x_pca[:,1]



color = ["red","green","blue"]

import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x_pca_ = pd.DataFrame(data=x_pca)

xs = x_pca_.iloc[5,0:10]

ys = x_pca_.iloc[5,10:20]

zs = x_pca_.iloc[5,20:30]



ax.scatter(xs,ys,zs)

ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')

ax.set_xlim3d(-50, 10)

ax.set_ylim3d(-20, 40)

ax.set_zlim3d(-20, 40)

plt.show()
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

for ar in range(1,len(x_pca_)):

    xs = x_pca_.iloc[ar,0:10]

    ys = x_pca_.iloc[ar,10:20]

    zs = x_pca_.iloc[ar,20:30]

    ax.scatter(xs,ys,zs)

    ax.set_xlabel('X Label')

    ax.set_ylabel('Y Label')

    ax.set_zlabel('Z Label')

    ax.set_xlim3d(-50, 10)

    ax.set_ylim3d(-20, 40)

    ax.set_zlim3d(-20, 40)

    plt.show()

    plt.pause(0.5)

    ax.cla()
from keras.models import Model

from keras.layers import Input, Dense

from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

import json, codecs

import warnings 

warnings.filterwarnings('ignore')

x = x.drop(['p1','p2'],axis=1)
input_img = Input(shape=(93,))



encoded = Dense(32, activation='relu')(input_img)



encoded = Dense(16, activation='relu')(encoded)



decoded = Dense(32, activation='relu')(encoded)



decoded = Dense(93, activation='relu')(decoded)



autoencoder = Model(input_img,decoded)



autoencoder.compile(optimizer='rmsprop',loss='binary_crossentropy')



hist = autoencoder.fit(x,x,epochs=200,batch_size=256,shuffle=True,validation_data=(x,x))



print(hist.history.keys())

plt.plot(hist.history['loss'],label = 'Train loss')

plt.plot(hist.history['val_loss'],label = 'Validationn loss')

plt.legend()

plt.show()
denemedix = autoencoder.predict(x)

denemedix = pd.DataFrame(data=denemedix)
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

for ar in range(1,len(denemedix)):

    xs = denemedix.iloc[ar,0:10]

    ys = denemedix.iloc[ar,10:20]

    zs = denemedix.iloc[ar,20:30]

    ax.scatter(xs,ys,zs)

    ax.set_xlabel('X Label')

    ax.set_ylabel('Y Label')

    ax.set_zlabel('Z Label')

    ax.set_xlim3d(-50, 10)

    ax.set_ylim3d(-20, 40)

    ax.set_zlim3d(-20, 40)

    plt.show()

    plt.pause(0.5)

    ax.cla()