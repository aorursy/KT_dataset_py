!tar xf /kaggle/input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz

# !ln -s /kaggle/input/fastai-audio/audio .

# !ln -s /kaggle/input/fastai/fastai .

# !pip install -qU "/kaggle/input/fastai2-wheels/fastprogress-0.2.2-py3-none-any.whl"
# !pip install -qU git+https://github.com/fastai/fastai

# !pip install -qU torch torchaudio torchvision

# !git clone https://github.com/mogwai/fastai_audio

# %cd fastai_audio

# !./install.sh

# %cd ../

# !ln -s /kaggle/input/fastai_audio/audio .
# from audio import *  

# from fastai.basics import *
# import gc

# from functools import partial

from pathlib import Path



import torchvision

from fastai.vision import *

from tqdm.notebook import tqdm





home = Path(".")

input_dir = Path("../input/deepfake-detection-challenge")
# from fastai.utils import *

# show_install(1)
# torch.__version__
# import torchvision

# torchvision.__version__
# torchaudio.__version__
# !apt-get --assume-yes install sox libsox-dev libsox-fmt-all libsndfile1
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



seed_everything(42)
labels = pd.read_json(input_dir/"train_sample_videos/metadata.json").T
ext = ".wav"
audio_path = Path("test_audio")

audio_path.mkdir(exist_ok=True)

train_audio_path = Path("train_audio")

train_audio_path.mkdir(exist_ok=True)
def mp4_to_wav(filenames, out):

    Path(out).mkdir(exist_ok=True)

    for fn in tqdm(filenames):

        out_fn = f"{out/fn.stem}{ext}"

        command = f"/kaggle/working/ffmpeg-git-20191209-amd64-static/ffmpeg -i '{fn}' -ar 44100 -vn '{out_fn}'"

        subprocess.call(command, shell=True)
# mp4_to_wav((input_dir/"test_videos").ls(), audio_path)
# mp4_to_wav((input_dir/"train_sample_videos").ls(), train_audio_path)
# config = AudioConfig()

# config.duration = 10_000
# get_y = lambda x: labels.loc[f"{x.stem}.mp4"].label
# audios = (AudioList.from_folder(train_audio_path, config=config)

#           .split_by_rand_pct(.2, seed=42)

#           .label_from_func(get_y))
BSA=1
# db = audios.databunch(bs=BSA)

# db.show_batch()
# learn = audio_learner(db)
# learn = load_learner("/kaggle/input/deepfake", "export.pkl")
# test = AudioList.from_folder(audio_path, config=config); test
# learn.predict(test[-1])
# preds = []

# for t in test:

#     preds.append(learn.data.classes[np.argmax(learn.predict(t)[2])])

# preds[:5]
# def predict_from_file(wav_file, learner, verbose=True):  

#     item = AudioItem(path=wav_file)

#     if verbose: display(item)

#     al = AudioList([item], path=item.path, config=config)

#     ai = AudioList.open(al, item.path)

#     y, pred, raw_pred = learner.predict(ai)

#     if verbose: print(y)

#     if verbose: print(pred.item())

#     if verbose: print(raw_pred)
# predict_from_file(test[0].path, learner )
# BS = 1 #CPU

BS = 864
!ln -s /kaggle/input/blazeface-pytorch/* ./

!ln -s /kaggle/input/deepfakes-inference-demo/helpers ./
from blazeface import BlazeFace

facedet = BlazeFace().to(torch.device("cuda:0"));

# facedet = BlazeFace();

facedet.load_weights("blazeface.pth")

facedet.load_anchors("anchors.npy")

_ = facedet.train(False)



from helpers.read_video_1 import VideoReader

from helpers.face_extract_1 import FaceExtractor



frames_per_video = 17



video_reader = VideoReader()

video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)

face_extractor = FaceExtractor(video_read_fn, facedet)
def extract_faces(video_path, batch_size):

    # Find the faces for N frames in the video.

    faces = face_extractor.process_video(video_path)



    # Only look at one face per frame.

    face_extractor.keep_only_best_face(faces)

    

    if len(faces) > 0:

        # NOTE: When running on the CPU, the batch size must be fixed

        # or else memory usage will blow up. (Bug in PyTorch?)

        x = []



        # If we found any faces, prepare them for the model.

        n = 0

        for frame_data in faces:

            for face in frame_data["faces"]:

                x.append(face)

                n += 1

    return x



def predict_on_mp4(names, learner, bs=17):

    preds = []

    for fn in tqdm(names):

        if fn.is_file():

            faces = extract_faces(fn, batch_size=bs)

            pred = [learner.predict(Image(torchvision.transforms.ToTensor()(f)))[2] for f in faces]

            if not pred:

                display(f"No pred from {fn}")

            preds.append(pred)

    return preds
def mace(pred:Tensor, targ:Tensor)->Rank0Tensor:

    "Mean absolute error between clamped `pred` and `targ`."

    pred,targ = flatten_check(pred,targ)

    return torch.abs(targ - pred.clamp(0., 1.)).mean()
learn = load_learner("/kaggle/input/deepfake", "float_shots.pkl", bs=BS).to_fp32()

# learn = load_learner("/kaggle/input/deepfake", "shots.pkl", bs=BS).to_fp32()
raw_preds = predict_on_mp4((input_dir/"test_videos").ls(), learn); len(raw_preds)
# # classes

# preds = []

# for p in raw_preds:

#     try:

#         preds.append(torch.stack(p, dim=0).argmax(dim=1).float().mean().item())

#     except:

#         preds.append(0.5)
preds = []

for p in raw_preds:

    try:

        preds.append(torch.stack(p, dim=0).mean().clamp(0.1,0.9).item())

    except:

        preds.append(0.5)
subm = pd.read_csv(input_dir/"sample_submission.csv")
subm["label"] = preds

# subm["label"] = 1 - subm["label"] # for binary classification where 0 is FAKE and 1 is REAL

subm["label"].value_counts(bins=2)
subm.to_csv("submission.csv", index=False, float_format='%.20f')
pd.read_csv("submission.csv")