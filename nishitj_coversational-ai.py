# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install simpletransformers
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from simpletransformers.conv_ai import ConvAIModel
from simpletransformers.conv_ai.conv_ai_utils import get_dataset
import torch
import random
train_args = {
    "overwrite_output_dir": True,
    "reprocess_input_data": True
}

# Create a ConvAIModel
conv_ai = ConvAIModel("gpt", "/kaggle/input/gpt_personachat_cache", use_cuda=False, args=train_args)
conv_ai.train_model()
model = conv_ai.model
args = conv_ai.args
tokenizer = conv_ai.tokenizer
process_count = conv_ai.args.process_count

conv_ai._move_model_to_device()
def get_personality(conv_ai, personality=None):
    if not personality:
        dataset = get_dataset(
            tokenizer,
            None,
            args.cache_dir,
            process_count=process_count,
            proxies=conv_ai.__dict__.get("proxies", None),
            interact=True,
            args=args,
        )
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        personality = random.choice(personalities)
    else:
        personality = [tokenizer.encode(s.lower()) for s in personality]
    return personality

def interact(conv_ai, raw_text, personality=None):
        """
        Interact with a model in the terminal.
        Args:
            personality: A list of sentences that the model will use to build a personality.
        Returns:
            None
        """
        history = []
        
        if not raw_text:
            return "Prompt should not be empty!"

        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = conv_ai.sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2 * args.max_history + 1) :]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_text
personality = get_personality(conv_ai)

interact(conv_ai, "hi, how are you", personality)
