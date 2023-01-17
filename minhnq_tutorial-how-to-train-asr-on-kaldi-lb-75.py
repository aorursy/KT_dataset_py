import os.path

import random



all_info = []

transcript = {}

path = '../input/train/audio/'



all_info = []

count = 0

# r=root, d=directories, f = files

for r, d, f in os.walk(path):

    for file in f:

        if '.wav' in file:

            count+=1

            trans = r.split("/")[-1]

            file_id = file.split(".")[0] + "_" + trans

            spk_id = file_id.split("_")[0]

            transcript[file_id] = trans

            all_info.append([spk_id,file_id,os.path.join(r, file)])



counter = int(len(all_info) * 0.1)

random.shuffle(all_info)

all_train_info = all_info[counter:]

all_test_info = all_info[:counter]
if not os.path.exists(os.path.dirname('data/train_command/text')):

    os.makedirs(os.path.dirname('data/train_command/text'))

if not os.path.exists(os.path.dirname('data/test_command/text')):

    os.makedirs(os.path.dirname('data/test_command/text'))



def text(file_infos):

    results = []

    # folder_path = os.path.abspath("recordings")

    for info in file_infos:

        utt_id = info[1]

        trans = transcript[utt_id]

        results.append("{} {}".format(utt_id, trans))

    return '\n'.join(sorted(results))



with open("data/train_command/text","wt") as f:

    f.writelines(text(all_train_info))

with open("data/test_command/text","wt") as f:

    f.writelines(text(all_test_info))
if not os.path.exists(os.path.dirname('data/train_command/wav.scp')):

    os.makedirs(os.path.dirname('data/train_command/wav.scp'))

if not os.path.exists(os.path.dirname('data/test_command/wav.scp')):

    os.makedirs(os.path.dirname('data/test_command/wav.scp'))



def wavscp(file_infos):

    results = []

    for info in file_infos:

        results.append("{} {}".format(info[1], info[2]))

    return '\n'.join(sorted(results))



with open("data/train_command/wav.scp","wt") as f:

    f.writelines(wavscp(all_train_info))

with open("data/test_command/wav.scp","wt") as f:

    f.writelines(wavscp(all_test_info))
if not os.path.exists(os.path.dirname('data/train_command/utt2spk')):

    os.makedirs(os.path.dirname('data/train_command/utt2spk'))

if not os.path.exists(os.path.dirname('data/test_command/utt2spk')):

    os.makedirs(os.path.dirname('data/test_command/utt2spk'))



def utt2spk(file_infos):

    results = []

    for info in file_infos:

        speaker = info[0]

        utt_id = info[1]

        results.append("{} {}".format(utt_id, speaker))

    return '\n'.join(sorted(results))



with open("data/train_command/utt2spk","wt") as f:

    f.writelines(utt2spk(all_train_info))

with open("data/test_command/utt2spk","wt") as f:

    f.writelines(utt2spk(all_test_info))
all_info = []

transcript = {}

path = '../input/data/test'



all_info = []

count = 0

# r=root, d=directories, f = files

for r, d, f in os.walk(path):

    for file in f:

        if '.wav' in file:

            count+=1

            file_name = file.split(".")[0]

            spk_id = file_name.split("_")[1]

            all_info.append([spk_id,spk_id + "_" + file_name,os.path.join(r, file)])





if not os.path.exists(os.path.dirname('data/eval_command/wav.scp')):

    os.makedirs(os.path.dirname('data/eval_command/wav.scp'))



def wavscp(file_infos):

    results = []

    for info in file_infos:

        results.append("{} {}".format(info[1], info[2]))

    return '\n'.join(sorted(results))



with open("data/eval_command/wav.scp","wt") as f:

    f.writelines(wavscp(all_info))





if not os.path.exists(os.path.dirname('data/eval_command/utt2spk')):

    os.makedirs(os.path.dirname('data/eval_command/utt2spk'))



def utt2spk(file_infos):

    results = []

    for info in file_infos:

        speaker = info[0]

        utt_id = info[1]

        results.append("{} {}".format(utt_id, speaker))

    return '\n'.join(sorted(results))



with open("data/eval_command/utt2spk","wt") as f:

    f.writelines(utt2spk(all_info))
if not os.path.exists(os.path.dirname('log/decode.1.log')):

    os.makedirs(os.path.dirname('log/decode.1.log'))

    

with open("log/decode.1.log","wt") as f:

    f.write("000044442_clip_000044442 no\n")

    f.write("LOG (gmm-latgen-faster[5.5.382~1-c2163]:DecodeUtteranceLatticeFaster():decoder-wrappers.cc:289) Log-like per frame for utterance 000044442_clip_000044442 is -3.39615 over 98 frames.\n")

    f.write("0000adecb_clip_0000adecb happy\n")

    f.write("LOG (gmm-latgen-faster[5.5.382~1-c2163]:DecodeUtteranceLatticeFaster():decoder-wrappers.cc:289) Log-like per frame for utterance 0000adecb_clip_0000adecb is -4.63466 over 98 frames.\n")
!cat log/decode.1.log
import re



all_info = []

path = 'log' #Path to log folder

eval = {}

count = 0

pattern = r'.{9}_clip_.{9}.*'



def _read_decode_file(filepath):

    with open(filepath, "rt") as f:

        for line in f.read().splitlines():

            if line.startswith("LOG"):

                continue

            x = re.search(pattern,line)

            if x is not None:

                res = x.group(0)

                info = res.split()

                utt_id = info[0]

                wav_id = utt_id[10:] + ".wav"

                if len(info) == 1:

                    trans = "silence"

                else:

                    trans = " ".join(info[1:])

                eval[wav_id] = trans

    pass





# r=root, d=directories, f = files

for r, d, f in os.walk(path):

    for file in f:

        if 'decode.' in file:

            count+=1

            _read_decode_file("/".join([r,file]))
all_lines=[]

with open("../input/sample_submission.csv","rt") as f:

    for line in f.read().splitlines():

        if line.startswith("fname"):

            all_lines.append(line)

            continue

        line = line.split(",")

        try:

            trans = eval[line[0]]

        except KeyError:

            trans = "silence"

        all_lines.append(",".join([line[0],trans]))



with open("submission.csv","wt") as f:

    f.writelines("\n".join(all_lines))