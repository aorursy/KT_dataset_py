import os
os.environ['WANDB_SILENT'] = 'True'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer
from scipy.special import softmax
MODEL_NAME = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def sentence_input(sentence: str, max_len: int = 512, device = 'cpu'):
    encoded = tokenizer.encode_plus(sentence, add_special_tokens=True, 
                                    pad_to_max_length=True, max_length=max_len, 
                                    return_tensors="pt",).to(device)
    model_input = (encoded['input_ids'], encoded['attention_mask'])
    return model_input
test_sentence = "Super Cute: First of all, I LOVE this product. When I bought it my husband jokingly said that it looked cute and small in the picture, but was really HUGE in real life. Don't tell him I said so, but he was right. It is huge and the cord is really long. Although I wish it was smaller, I still love it. It works really well when we travel and need to plug a lot of things in and although the length is annoying, it's very useful."
model_input = sentence_input(test_sentence)
print(test_sentence)
print(model_input)
import torch.nn as nn
import torch
class DistilBert(nn.Module):
    """
    Simplified version of the same class by HuggingFace.
    See transformers/modeling_distilbert.py in the transformers repository.
    """

    def __init__(self, pretrained_model_name: str, num_classes: int = None):
        """
        Args:
            pretrained_model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_classes (int): the number of class labels
                in the classification task
        """
        super().__init__()

        config = AutoConfig.from_pretrained(
             pretrained_model_name)

        self.distilbert = AutoModel.from_pretrained(pretrained_model_name,
                                                    config=config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, features, attention_mask=None, head_mask=None):
        """Compute class probabilities for the input sequence.

        Args:
            features (torch.Tensor): ids of each token,
                size ([bs, seq_length]
            attention_mask (torch.Tensor): binary tensor, used to select
                tokens which are used to compute attention scores
                in the self-attention heads, size [bs, seq_length]
            head_mask (torch.Tensor): 1.0 in head_mask indicates that
                we keep the head, size: [num_heads]
                or [num_hidden_layers x num_heads]
        Returns:
            PyTorch Tensor with predicted class probabilities
        """
        assert attention_mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(input_ids=features,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        # we only need the hidden state here and don't need
        # transformer output, so index 0
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        # we take embeddings from the [CLS] token, so again index 0
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits
from transformers import AutoConfig, AutoTokenizer, AutoModel
model = DistilBert(pretrained_model_name=MODEL_NAME,
                                           num_classes=2)
from catalyst.dl.utils import trace
def load_chechpoint(model, path):
    mod = trace.load_checkpoint(path)
    model.load_state_dict(mod['model_state_dict'])
    return model


model = load_chechpoint(model, '../input/sentiment-all-models/last 0.9622.pth')

model.eval()

traced_cpu = torch.jit.trace(model, model_input)
torch.jit.save(traced_cpu, "cpu.pth")

#to load
cpu_model = torch.jit.load("cpu.pth")

# GPU
# traced_gpu = torch.jit.trace(model.cuda(), gpu_model_input)
# torch.jit.save(traced_gpu, "gpu.pth")

# gpu_model = torch.jit.load("gpu.pth")
print(cpu_model.graph)
quantized_model = torch.quantization.quantize_dynamic(model)
print(quantized_model)
!pip install onnx onnxruntime onnxruntime-tools
#For GPU Inference: install onnxruntime-gpu
torch.onnx.export(model, model_input, "model_512.onnx",
                  export_params=True,
                  input_names=["input_ids", "attention_mask"],
                  output_names=["targets"],
                  dynamic_axes={
                      "input_ids": {0: "batch_size"},
                      "attention_mask": {0: "batch_size"},
                      "targets": {0: "batch_size"}
                  },
                  verbose=True)
import onnx
onnx_model = onnx.load('model_512.onnx')
onnx.checker.check_model(onnx_model, full_check=True)
onnx.helper.printable_graph(onnx_model.graph)
from onnxruntime_tools import optimizer
optimized_model_512 = optimizer.optimize_model("model_512.onnx", model_type='bert', 
                                               num_heads=12, hidden_size=768,
                                              use_gpu=False, opt_level=99)

optimized_model_512.save_model_to_file("optimized_512.onnx")
import onnxruntime as ort
print(ort.get_device())
OPTIMIZED_512 = ort.InferenceSession('./optimized_512.onnx')
def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

def prediction_onnx(model, sentence: str, max_len: int = 512):
    encoded = tokenizer.encode_plus(sentence, add_special_tokens=True, 
                                    pad_to_max_length=True, max_length=max_len,
                                    return_tensors="pt",)
    # compute ONNX Runtime output prediction
    input_ids = to_numpy(encoded['input_ids'])
    attention_mask = to_numpy(encoded['attention_mask'])
    onnx_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    logits = model.run(None, onnx_input)
    preds = softmax(logits[0][0])
    print(f"Class: {['Negative' if preds.argmax() == 0 else 'Positive'][0]}, Probability: {preds.max():.4f}")
prediction_onnx(OPTIMIZED_512, test_sentence)
df = pd.DataFrame([[506, 273, 151, 89.1, 0],
                  [507, 263, 145, 82.7, 5.2],
                  [516, 237, 126, 72.4, 19],
                  [388, 180, 92.2, 49.7, 56.2]], index = ['Pytorch', 'TorchScript', 
                                                    'ONNX Runtime', 'Quantization'],
                  columns=['512', '256', '128', '64', "Av.SpeedUp (%)"])
display(df)
cpu = pd.DataFrame([[64, 'Pytorch', 89.1],
                  [64, 'TorchScript', 82.7],
                  [64, 'ONNX Runtime', 72.4],
                  [64, 'Quantization', 49.4],
                   [128, 'Pytorch', 151],
                   [256, 'Pytorch', 273],
                   [512, 'Pytorch', 506],
                   [128, 'TorchScript', 145],
                   [256, 'TorchScript', 263],
                   [512, 'TorchScript', 507],
                   [128, 'ONNX Runtime', 126],
                   [256, 'ONNX Runtime', 237],
                   [512, 'ONNX Runtime', 516],
                   [128, 'Quantization', 92.2],
                   [256, 'Quantization', 180],
                    [512, 'Quantization', 388]],
                  columns=['Sequence', 'Optimization', 'Time (ms)'])

sns.set_style("darkgrid")
sns.catplot(x='Optimization', y='Time (ms)', data=cpu, kind='bar',
            ci=None, col='Sequence', col_wrap=2,
           col_order=[512,256,128,64]);
gpu_df = pd.DataFrame([[16.1, 12.1, 11.9, 11.9, 0],
                  [15.9, 11.2, 9.2, 8.92, 18],
                  [14.2, 10, 8.14, 7.57, 35]], index = ['Pytorch', 'TorchScript', 
                                                    'ONNX Runtime'],
                  columns=['512', '256', '128', '64', "Av.SpeedUp (%)"])
display(gpu_df)
gpu = pd.DataFrame([[64, 'Pytorch', 11.9],
                  [64, 'TorchScript', 8.92],
                  [64, 'ONNX Runtime', 7.57],
                   [128, 'Pytorch', 11.9],
                   [256, 'Pytorch', 12.1],
                   [512, 'Pytorch', 16.1],
                   [128, 'TorchScript', 9.2],
                   [256, 'TorchScript', 11.2],
                   [512, 'TorchScript', 15.9],
                   [128, 'ONNX Runtime', 8.14],
                   [256, 'ONNX Runtime', 10],
                   [512, 'ONNX Runtime', 14.2]],
                  columns=['Sequence', 'Optimization', 'Time (ms)'])

sns.catplot(x='Optimization', y='Time (ms)', data=gpu, kind='bar',
            ci=None, col='Sequence', col_wrap=2,
           col_order=[512,256,128,64]);