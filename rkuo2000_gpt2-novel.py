from IPython.display import clear_output
!pip install pytorch-transformers
!pip install tqdm
!pip install torchsnooper
clear_output()
import sys
import json
import torch
import textwrap
import torchsnooper
import pytorch_transformers
import torch.nn.functional as F
from tqdm import trange
from IPython.core.display import display, HTML
!git clone https://github.com/zjcai96/bertviz bertviz_repo
!pip install regex
clear_output()
GITHUB_REPO = "GPT2-Chinese"
!rm -rf {GITHUB_REPO}
!git clone https://github.com/Morizeyao/{GITHUB_REPO}.git {GITHUB_REPO}
if not GITHUB_REPO in sys.path:
    sys.path += [GITHUB_REPO]
!ls GPT2-Chinese
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Z8WdVYgBj01BHU4syjlY9qj3KBfEFP2D' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Z8WdVYgBj01BHU4syjlY9qj3KBfEFP2D" -O 10layers_12heads_1024len_768embd_full_corpus_16bsize.zip && rm -rf /tmp/cookies.txt
pretrained_model = '10layers_12heads_1024len_768embd_full_corpus_16bsize'
!unzip {pretrained_model}.zip
sagemaker_base_path = 'home/ec2-user/SageMaker/tmp/GPT2-Chinese'
config_file = 'config.json'
model_ckpt = "pytorch_model.bin"
vocab_file = "vocab_small.txt"

!rm {config_file} {model_ckpt} {vocab_file}

!mv {sagemaker_base_path}/model/{pretrained_model}/{config_file} {config_file}
!mv {sagemaker_base_path}/model/{pretrained_model}/{model_ckpt} {model_ckpt}
!mv {sagemaker_base_path}/cache/{vocab_file} {vocab_file}
from tokenizations import tokenization_bert

# make model output attentions
config = pytorch_transformers.GPT2Config.from_json_file(config_file)
config.output_attentions = True


model = pytorch_transformers.GPT2LMHeadModel.from_pretrained(".", config=config)
tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
clear_output()
def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def fast_sample_sequence(model,context,length,temperature=1, top_k=0, top_p=0.0,device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            
            # redraw if [UNK]
            if next_token.unsqueeze(0) != 100:
                generate.append(next_token.item())
                prev = next_token.view(1, 1)

    return generate


def get_html(context, generated_text, novel_name='', algorithm=''):
    if generated_text[-1] != '。':
        if generated_text[-1] == '，':
            generated_text= generated_text[:-1]
        generated_text += ' ...'
    
    html = """
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * {
      box-sizing: border-box;
    }

    /* Create two unequal columns that floats next to each other */
    .column {
      float: left;
      padding: 10px;
    }

    .left {
      width: 12%;
    }

    .right {
      width: 35%;
    }

    /* Clear floats after the columns */
    .row:after {
      content: "";
      display: table;
      clear: both;
    }
    </style>
    </head>
    <body>

    <div style="background-color:#404452 !important">
        <div class="row">
          <div class="column left">
            <h2 style="color: #bfbaba;text-align:center">
                novel_name
                前文脈絡
            </h2>
          </div>
          <div class="column right">
            <h3 style="color: white;line-height: 1.5">context</h3>
          </div>
        </div>

        <hr/>

        <div class="row">
          <div class="column left">
            <h2 style="color: #bfbaba;text-align:center">
                algorithm
                生成結果
            </h2>
          </div>
          <div class="column right">
            <h3 style="color: white;line-height: 1.5">generated_text</h3>

          </div>
        </div>
    </div>

    </body>
    </html>


    """.replace('context', context).replace('generated_text', generated_text).replace("novel_name", f'《{novel_name}》<br/>')
    
    if not algorithm:
        html = html.replace("algorithm", "")
    else:
        html = html.replace("algorithm", f'{algorithm}<br/>')
    
    
    return html


def generate(context, topk, topp, temperature, device, line_len=40, novel_name=''):
    context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
        
    # auto-regressive
    out = fast_sample_sequence(
        model=model, length=length,
        context=context_tokens,
        temperature=temperature, top_k=topk, top_p=topp, device=device
    )

    # rendering
    tokens = tokenizer.convert_ids_to_tokens(out)

    for i, item in enumerate(tokens):
        if item == '[MASK]':
            tokens[i] = ''
        if item == '[CLS]' or item == '[SEP]':
            tokens[i] = '\n'
    
    generated_text = ''.join(tokens).strip().replace(context, '')
    html = get_html(context, generated_text, novel_name=novel_name)
    
    return html, generated_text
nsamples = 1
batch_size = 1
length = model.config.n_ctx // 2

topk = 30
topp = 0
temperature = 1
# select index of sampled_contexts
sample_idx = 3 


# 飛雪連天射白鹿，笑書神俠倚碧鴛。
sampled_contexts =[
    ('飛狐外傳', '胡斐行動快極，右手彎處，抱住了程靈素的纖腰，倒縱出門，經過房門時飛起一腿，踢在門板之上。'),
    ('雪山飛狐', '胡一刀抱著孩子走進房去，那房間的板壁極薄，只聽夫人問道：‘大哥，是誰來了啊？’'),
    ('連城訣', '戚芳躲在狄雲背後，也不見禮。只點頭笑了笑。'),
    ('天龍八部', '段譽和王語嫣面面相對，呼吸可聞，雖身處污泥，心中卻充滿了喜樂之情，誰也沒想到要爬出井去。兩人同時慢慢的伸手出來，四手相握，心意相通。'),
    ('射雕英雄傳', '黃蓉眼圈兒一紅，道：「爹爹不要我啦。」郭靖道：「乾麼呀？」'),
    ('白馬嘯西風', '史仲俊和白馬李三的妻子上官虹原是同門師兄妹，兩人自幼一起學藝。'),
    ('鹿鼎記', '韋小寶只覺滿鼻子都是濃香，懷中抱著的那女子全身光溜溜地，竟然一絲不掛，又覺那女子反手過來，抱住了自己，心中一陣迷迷糊糊，聽得雙兒低聲問道：「相公，怎麼了？」韋小寶唔唔幾聲，待要答話，懷中那女子伸嘴吻住了他嘴巴，登時說不出話來。'),
    ('笑傲江湖', '令狐沖淡淡一笑，道：「原要多謝兩位的救命之恩。」王家駒聽他語氣，知他說的乃是反話，更加有氣，大聲道：「你是華山派掌門大弟子，連洛陽城中幾個流氓混混也對付不了，嘿嘿，旁人不知，豈不是要說你浪得虛名？」'),
    ('書劍恩仇錄', '陳家洛在下首站定，微一拱手，說道：「請賜招。」'),
    ('神鵰俠侶', '黃蓉見楊過中毒極深，低聲道：「咱們先投客店，到城裡配幾味藥。」'),
    ('俠客行', '石破天見茶几上放著兩碗清茶，便自己左手取了一碗，右手將另一碗遞過去。陳衝之既怕茶中有毒，又怕石破天乘機出手，不敢伸手去接，反退了一步，嗆啷一聲，一隻瓷碗在地下摔得粉碎。'),
    ('倚天屠龍記', '張無忌見三名老僧在片刻間連斃崑崙派四位高手，舉重若輕，游刃有餘，武功之高，實是生平罕見，比之鹿杖客和鶴筆翁似乎猶有過之，縱不如太師父張三丰之深不可測，卻也到了神而明之的境界。'),
    ('碧血劍', '張朝唐聽到這裡，才知道這神像原來是連破清兵、擊斃清太祖努爾哈赤、使清人聞名喪膽的薊遼督師袁崇煥。'),
    ('鴛鴦刀', '蕭中慧一聽父親說起這對寶刀，當即躍躍欲試。'),
    ('天龍八部', '蕭峯喝道：「你就想走？天下有這等便宜事？你父親身上有病，大丈夫不屑乘人之危，且放了他過去。你可沒病沒痛！」慕容復氣往上衝，喝道：「那我便接蕭兄的高招。」蕭峯更不打話，呼的一掌，一招降龍十八掌中的「見龍在田」，向慕容復猛擊過去。他見藏經閣中地勢狹隘，高手群集，不便久鬥，是以使上了十成力，要在數掌之間便取了敵人性命。慕容復見他掌勢兇惡，當即運起平生之力，要以「斗轉星移」之術化解。'),
]

sample = sampled_contexts[sample_idx]
novel_name, context = sample
html, generated_text = generate(context, topk, topp, temperature, device, novel_name=novel_name)
display(HTML(html))
!ln -s bertviz_repo/bertviz bertviz 
def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
      '''))
    

from bertviz.pytorch_transformers_attn.modeling_gpt2 import GPT2Model
from bertviz.head_view_gpt2 import show as show_head
from bertviz.model_view_gpt2 import show as show_model
from bertviz.neuron_view_gpt2 import show as show_neuron


def call_html(view):
    import IPython
    if view in ['model', 'neuron']:
        display(IPython.core.display.HTML('''
             <script src="/static/components/requirejs/require.js"></script>
             <script>
               requirejs.config({
                 paths: {
                   base: '/static/base',
                   "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min",
                   jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
                 },
               });
             </script>
        '''))
    else:
        display(IPython.core.display.HTML('''
             <script src="/static/components/requirejs/require.js"></script>
             <script>
               requirejs.config({
                 paths: {
                   base: '/static/base',
                   "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
                   jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
                 },
               });
             </script>
       '''))
  

def show(model, tokenizer, text, view, model_type='gpt2'):
    call_html(view)
    show_func = {'head': show_head, 'model': show_model, 'neuron': show_neuron}
    show_func[view](model, tokenizer, text)
    

gpt2_model = GPT2Model.from_pretrained('.')
gpt2_model.to('cpu')
gpt2_model.eval()
clear_output()
text = '喬峯帶阿朱回到北方，喬峯對她說：「我們兩人永遠留在這裡！」'

view = 'head'
show(gpt2_model, tokenizer, text, view)
#view = 'neuron'
#show(gpt2_model, tokenizer, text, view)
#view = 'model'
#show(gpt2_model, tokenizer, text, view)
