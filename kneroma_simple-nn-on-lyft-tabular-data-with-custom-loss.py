import pandas as pd, numpy as np

import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

import re,json,time,pickle

from sklearn.preprocessing import MinMaxScaler
pd.options.display.max_columns=305
DATA_ROOT = Path("../input/lyft-train-as-parquet/train")
scaler = MinMaxScaler()
def get_scene_path(scene):

    meta = "meta_{}_{}.json".format(*re.search( r"scenes_(\d+)_(\d+)", scene.stem).groups())

    with open(DATA_ROOT/meta) as f:

        meta = json.load(f)

    frame = DATA_ROOT/meta["frames"]["results"]["filename"]

    agent = DATA_ROOT/meta["agents"]["results"]["filename"]

    return (scene, frame, agent)
SCENES = np.array(list(DATA_ROOT.glob("scenes_*.parquet.snappy")))

SCENES = SCENES[np.random.permutation(len(SCENES))]

print("NB SCENES:", len(SCENES))

scene = SCENES[0]

scene
get_scene_path(scene)
reader = pd.read_parquet
TRAIN_COLS = [

#     'ego_translation_x', 

#     'ego_translation_y', 

#     'ego_translation_z', 

#     'ego_rotation_xx', 

#     'ego_rotation_xy', 

#     'ego_rotation_xz', 

#     'ego_rotation_yx', 

#     'ego_rotation_yy', 

#     'ego_rotation_yz', 

#     'ego_rotation_zx', 

#     'ego_rotation_zy', 

#     'ego_rotation_zz', 

    'extent_x_shift_50', 

    'extent_y_shift_50', 

    'extent_z_shift_50', 

    'velocity_x_shift_50', 

    'velocity_y_shift_50', 

    'label_probabilities_PERCEPTION_LABEL_UNKNOWN_shift_50', 

    'label_probabilities_PERCEPTION_LABEL_CAR_shift_50', 

    'label_probabilities_PERCEPTION_LABEL_CYCLIST_shift_50', 

    'label_probabilities_PERCEPTION_LABEL_PEDESTRIAN_shift_50', 

    'yaw_shift_50', 

    'nagents_shift_50', 

    'nlights_shift_50', 

    'centroid_x_shift_50', 

    'centroid_y_shift_50',

]
def read_list(df_names):

    df = None

    for df_name in df_names:

        temp = reader(DATA_ROOT/df_name)

        df = df.append(temp) if df is not None else temp

        

    return df
def get_shifted_col_names(cols, shift):

    new_cols = []

    for col in cols:

        col = re.sub(r"_shift_(\d\d$)", "", col)

        new_col = f"{col}_shift_{shift:02d}"

        new_cols.append(new_col)

    return new_cols
def df_shifter(group_cols, shift_cols,  max_shift=1, shifts=None, keep_nan=False, verbose=0):

    global df

    assert max_shift >= 1

    cols = None

#     df = globals()["df"]

    shifts = shifts or range(1, max_shift+1)

    for ishift in shifts:

        if cols is not None:

            shift_cols  = cols

#         cols = ["{}_shift_{:02d}".format(col, ishift) for col in shift_cols]

        cols = get_shifted_col_names(shift_cols, shift=ishift)

        

        df[cols] = df.groupby(group_cols)[shift_cols].shift()

        

        if not keep_nan:

            df = df[df[cols].notnull().all(1)]

        if verbose:

            print("ishift: {}  df.shape: {}".format(ishift, df.shape))

#             del globals()["df"]

#             del df

    

    df.rename(columns={"centroid_x": "centroid_x_shift_00", "centroid_y": "centroid_y_shift_00"}, inplace=True)

    return df
def merge(scenes, frames, agents, max_shift=50, verbose=False):

    global df

    df = scenes.merge(frames, on = "scene_db_id")

    df = df.merge(agents, on="frame_db_id")



    df["nframes"] = df.groupby(["scene_db_id", "track_id"])["scene_db_id"].transform("count")

    df = df[df["nframes"] > max_shift]

    

    shift_cols = [

#             "centroid_x", "centroid_y",

            "yaw",

            "velocity_x","velocity_y",

            "nagents","nlights",

            'extent_x', 'extent_y','extent_z',

            'label_probabilities_PERCEPTION_LABEL_UNKNOWN',

            'label_probabilities_PERCEPTION_LABEL_CAR', 

            'label_probabilities_PERCEPTION_LABEL_CYCLIST', 

            'label_probabilities_PERCEPTION_LABEL_PEDESTRIAN',

    ]

    

    df[shift_cols] = df[shift_cols].astype(np.float32, copy=True)

    df[["scene_db_id", "track_id"]] = df[["scene_db_id", "track_id"]].astype(np.float32, copy=True)



    shape0 =  df.shape

    df = df_shifter(group_cols=["scene_db_id", "track_id"], shift_cols=shift_cols, shifts=[max_shift],

                        keep_nan=True, verbose=verbose)

    df = df_shifter(group_cols=["scene_db_id", "track_id"], shift_cols=["centroid_x_shift_00","centroid_y_shift_00"],

                    max_shift=max_shift, keep_nan=False, verbose=verbose)

    

    if verbose:

        print("SHAPE0:", shape0)

        print("SHAPE1:", df.shape)

        print("Nulls ratio:", 1-len(df)/shape0[0])

    

    return df
def read_all(max_shift=50, max_len=5e6, verbose=0):

    dfs = None

    scenes = SCENES[np.random.permutation(len(SCENES))]

    for scene in scenes:

        SCENES_FILE,FRAMES_FILE,AGENTS_FILE = get_scene_path(scene)

        

        scenes = reader(SCENES_FILE)

        

        frames = reader(FRAMES_FILE)

#         frames["frame_rank"] = frames.groupby("scene_db_id").scene_db_id.cumcount()

        frames["nagents"] = frames["agent_index_interval_end"] - frames["agent_index_interval_start"]

        frames["nlights"] = frames["traffic_light_faces_index_interval_end"

                                  ] - frames["traffic_light_faces_index_interval_start"]

    

        agents = reader(AGENTS_FILE)

        agents.rename(columns = {"agent_id": "agent_db_id"}, inplace=True)

        

        df = merge(scenes, frames, agents, max_shift=max_shift, verbose=verbose)

        

        dfs = df if dfs is None else dfs.append(df)

        dfs.reset_index(inplace=True, drop=True)

        

        if len(dfs) > max_len:

            break

    

    return dfs
%%time



# Increase max_len  for better results --> memory overflow risk !!!

# Locally, I used max_len=12e6

df = read_all(max_shift=50, verbose=0, max_len=1e5)
df.isnull().any(1).sum()/len(df)
df.columns
temp = sorted(df.columns[df.columns.str.startswith("centroid_") & ~df.columns.isin(TRAIN_COLS)])

XTARGET_COLS = [col for col in temp if "_x_" in col][::-1]

YTARGET_COLS  = [col for col in temp if "_y_" in col][::-1]

XTARGET_COLS,YTARGET_COLS
%%time



X  = scaler.fit_transform(df[TRAIN_COLS].values.astype("float32", copy=False))

TARGET = np.stack([

    df[XTARGET_COLS].values.astype("float32", copy=False) - df[["centroid_x_shift_50"]].values.astype("float32"),

    df[YTARGET_COLS].values.astype("float32", copy=False) - df[["centroid_y_shift_50"]].values.astype("float32"),

],

    axis=1,

)

X.shape, TARGET.shape
import torch

from torch import nn, optim
class SimpleNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(len(TRAIN_COLS), 124), nn.ReLU(), nn.Dropout(0.2),

            nn.Linear(124, 512), nn.ReLU(), nn.Dropout(0.2),

            nn.Linear(512, 2048), nn.ReLU(),nn.Dropout(0.2),

            nn.Linear(2048, 1024), nn.ReLU(),nn.Dropout(0.2),

        )

        

        self.xynet = nn.Linear(1024, 300)

        

        self.cnet = nn.Sequential(

            nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.2),

            nn.Linear(512, 3),

        )

        

    def forward(self, x):

        features = self.net(x)

        xy = self.xynet(features)

        c = self.cnet(features)

        

        return c,xy


def shapefy(xy_pred, xy):

    NDIM = 3

    xy_pred = xy_pred.view((-1,2, NDIM, 50))

    xy = xy[:,:, None].repeat([1,1, NDIM, 1])

    return xy_pred, xy



def LyftLoss(c, xy_pred, xy):

    xy_pred, xy  = shapefy(xy_pred, xy)

    

    c = torch.softmax(c, dim=1)

    

    l = torch.sum(torch.square(xy_pred-xy), dim=(1,3))/2

    

    # The LogSumExp trick for better numerical stability

    # https://en.wikipedia.org/wiki/LogSumExp

    m = l.min(dim=1).values

    l = torch.exp(m[:, None]-l)

    

    l = m - torch.log(torch.sum(l*c, dim=1))

    l = torch.mean(l)

    return l





def MSE(xy_pred, xy):

    xy_pred, xy = shapefy(xy_pred, xy)

    return torch.mean(torch.sum(torch.square(xy_pred-xy), 3))



def MAE(xy_pred, xy):

    xy_pred, xy = shapefy(xy_pred, xy)

    return torch.mean(torch.sum(torch.abs(xy_pred-xy), 3))
train_set, valid_set = train_test_split(np.arange(len(X)).reshape((-1,1)),

                                           np.arange(len(X)), test_size = .20, random_state=177)[2:]

len(train_set), len(valid_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = SimpleNet().to(device)

criterion = LyftLoss

optimizer = optim.Adam(net.parameters(), lr = 5e-4)
%%time



sel = np.arange(len(train_set))

batch_size = 5000

K = len(sel)//(5*batch_size)

EPOCHS = 1 # For demo only, please choose a right value by yourself, You may need to enable GPU



for epoch  in tqdm(list(range(EPOCHS))) :

    net.train()

    np.random.shuffle(sel)

    l,mse,mae, icount = 0.,0.,0., 0

    ibatch = 0

    for ibatch in tqdm(list(range(0, len(sel), batch_size)), leave = ibatch >= len(sel) - batch_size) :

        s = sel[ibatch:ibatch+batch_size]

        xb, yb = torch.from_numpy(X[s]).to(device), torch.from_numpy(TARGET[s]).to(device)

        

        optimizer.zero_grad()

        c,o = net(xb)

        loss = criterion(c, o, yb)

        loss.backward()

        optimizer.step()

        

        with torch.no_grad():

            l += loss.item()

            mse += MSE(o,yb).item()

            mae += MAE(o,yb).item()

        

        icount += 1

        

        if not (icount)%K:

            with torch.no_grad():

                s_valid = np.random.choice(valid_set, 100000, replace = False)

                l_valid, mse_valid, mae_valid, valid_count = 0.,0.,0., 0

                b = 10000

                for i_valid in range(0, len(s), b):

                    s = s_valid[i_valid:i_valid+b]

                    xb, yb = torch.from_numpy(X[s]).to(device), torch.from_numpy(TARGET[s]).to(device)



                    c,o = net(xb)

                    l_valid += criterion(c, o, yb)

                    mse_valid += MSE(o,yb).item()

                    mae_valid += MAE(o,yb).item()



                    valid_count += 1

                print("[{}-{}]  loss: ({:0.5f}, {:0.5f})  rmse: ({:0.5f}, {:0.5f})  mae: ({:0.5f}, {:0.5f})".format(

                    epoch,ibatch, l/K, l_valid/valid_count, np.sqrt(mse/K), np.sqrt(mse_valid/valid_count),

                mae/K, mae_valid/valid_count))

                

                l,mse,mae, icount = 0.,0.,0., 0
torch.save(net.state_dict(), "simple_net_00.pth")

with open("scaler_simple_net_00.bin", "wb") as f:

    pickle.dump(scaler, f)
# Loading locally pretrained weights and scaler

net = SimpleNet().to(device)

net.load_state_dict(torch.load("../input/neural-net-on-lyft-tabular-data/simple_net_00.pth", map_location=device))

net = net.eval()



with open("../input/neural-net-on-lyft-tabular-data/scaler_net_00.bin", "rb") as f:

    scaler = pickle.load(f)
# %%time



df_sub = pd.read_csv("../input/lyft-test-set-as-csv/Lyft_test_set.csv")

df_sub.rename(

    columns=dict(zip([col.replace("_shift_50", "") for col in TRAIN_COLS], TRAIN_COLS)), inplace=True

)

print("df_test:", df_sub.shape)

df_sub.head(10)
X_sub = df_sub[TRAIN_COLS].values.astype("float32")

X_sub = scaler.transform(X_sub)

X_sub.shape
def make_colnames():

    xcols = ["coord_x{}{}".format(step, rank) for step in range(3) for rank in range(50)]

    ycols = ["coord_y{}{}".format(step, rank) for step in range(3) for rank in range(50)]

    cols = ["timestamp", "track_id"] + ["conf_0", "conf_1", "conf_2"] + xcols + ycols

    return cols
%%time



b=1000

preds = []

cs = []

with torch.no_grad(): 

    for  icount in tqdm(list(range(0, len(X_sub), b))):

        xb = torch.from_numpy(X_sub[icount:icount+b])

        c, yb = net(xb)

        c = torch.softmax(c, dim=1)

        cs.append(c.cpu().numpy())

        preds.append(yb.cpu().numpy())

preds = np.vstack(preds)

cs = np.vstack(cs)

preds.shape, cs.shape
cols = make_colnames()

sub = pd.DataFrame(np.hstack([cs, preds]), columns = cols[2:])

sub[["timestamp", "track_id"]] = df_sub[["timestamp", "track_id"]].astype(int)

sub = sub[cols]

print("sub.shape:", sub.shape)

sub.head(10)
%%time



sub.to_csv("submission.csv", index=False, float_format="%.5f")