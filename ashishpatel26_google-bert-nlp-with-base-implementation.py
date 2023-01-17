# import sys

# !test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
# if not 'bertviz_repo' in sys.path:
#     sys.path += ['bertviz_repo']
# from bertviz import attention, visualization
# from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
# // %%javascript
# // require.config({
# //   paths: {
# //       d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min'
# //   }
# // });
# def call_html():
#     import IPython
#     display(IPython.core.display.HTML('''
#         <script src="/static/components/requirejs/require.js"></script>
#         <script>
#           requirejs.config({
#             paths: {
#               base: '/static/base',
#               "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
#               jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
#             },
#           });
#         </script>
#         '''))
# bert_version = 'bert-base-uncased'
# model = BertModel.from_pretrained(bert_version)
# tokenizer = BertTokenizer.from_pretrained(bert_version)
# sentence_a = "I went to the mall."
# sentence_b = "At the mall, I bought clothes for me."
# attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)
# tokens_a, tokens_b, attn = attention_visualizer.get_viz_data(sentence_a, sentence_b)
# call_html()
# attention.show(tokens_a, tokens_b, attn)