!pip freeze > kaggle_image_requirements.txt
# Pipeline uses `gpt2` by default, but we specify it explicitly to be fully transparent
from transformers import pipeline
gpt = pipeline('text-generation',model='gpt2')
gpt("Transfer learning is a field of study", max_length=100)
from transformers import AutoModelWithLMHead, AutoTokenizer # you can use these utility classes that automatically load the right classes
from transformers import GPT2LMHeadModel, GPT2Tokenizer # or these more specific classes directly
import torch

tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
# Chat for 5 Lines
conversation_length = 5
for step in range(conversation_length):
    # encode new user input, add end-of-sentence token, return tensor
    new_user_inputs_ids = tokenizer.encode(input("User: ") + tokenizer.eos_token, return_tensors='pt')
    
    # add new input to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_inputs_ids], dim=1) if step > 0 else new_user_inputs_ids
    
    # generate a response of up to max_length tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # display response
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))