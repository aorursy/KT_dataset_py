# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.chdir("../input")

from tf2bert import baseline, modeling, optimization , tokenization



os.chdir("../working")
bert_config = modeling.BertConfig.from_json_file("../input/bertjointbaseline/bert_config.json")



tokenizer = tokenization.FullTokenizer(vocab_file="../input/bertjointbaseline/vocab-nq.txt", do_lower_case=True)



learning_rate = 5e-5



otpimizer = optimization.create_optimizer(init_lr=learning_rate, num_train_steps=num_train_steps, num_warmup_steps=100)



model_fn = baseline.model_fn_builder(

    bert_config=bert_config,

    init_checkpoint="../input/bertjointbaseline/bert_joint.ckpt",

    learning_rate=learning_rate,

    num_train_steps=None,

    num_warmup_steps=None)