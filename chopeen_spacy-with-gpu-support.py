# CUDA

!nvcc --version
# CuPy

import cupy

cupy.show_config()
!pip install --upgrade --quiet spacy[cuda101]
!pip install --quiet https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
from pathlib import Path



DIR_DATA_INPUT = Path("../input/githubcord19/data/raw/")

FILE_RF_SENTENCES = DIR_DATA_INPUT / "cord_19_rf_sentences.jsonl"

FILE_ABSTRACTS_FILTERED = DIR_DATA_INPUT / "cord_19_abstracts_filtered.jsonl"

FILE_ABSTRACTS = DIR_DATA_INPUT / "cord_19_abstracts.jsonl"



DIR_MODELS = Path("/kaggle/working/models")

DIR_MODELS_RF_SENT = DIR_MODELS / "tok2vec_rf_sent_sci"

DIR_MODELS_ABS_FIL = DIR_MODELS / "tok2vec_abs_fil_sci"

DIR_MODELS_ABS = DIR_MODELS / "tok2vec_abs_sci"
!rm -rf $DIR_MODELS

!mkdir $DIR_MODELS
# RF sentences

!spacy pretrain $FILE_RF_SENTENCES en_core_sci_lg $DIR_MODELS_RF_SENT --use-vectors --n-iter 300
# filtered abstracts

!spacy pretrain $FILE_ABSTRACTS_FILTERED en_core_sci_lg $DIR_MODELS_ABS_FIL --use-vectors --n-iter 300
# all abstracts

!spacy pretrain $FILE_ABSTRACTS en_core_sci_lg $DIR_MODELS_ABS --use-vectors --n-iter 300