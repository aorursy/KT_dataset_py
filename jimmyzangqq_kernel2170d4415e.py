!python ../input/pylizhi-py/train.py --model_config ../input/pylizhi1/models/model_config.json --output_dir ../working  --tokenizer_path ../input/pylizhi1/models/vocab.txt --raw_data_path ../input/pylizhi1/data/cnews.json --batch_size 2 --pretrained_model ../input/pylizhi6/model_epoch1/ --tokenized_data_path ../input/tokenized/tokenized/ --epochs 1
!python ../input/pylizhi1/generation2.py --prefix 我面试总是不能通过，没有办法找到工作 --length 400 --nsamples 4 --model_config ../input/pylizhi6/model_epoch1/config.json --model_path ../input/pylizhi6/model_epoch1  --tokenizer_path ../input/pylizhi1/models/vocab.txt --temperature 1 --topk 90 --topp 8 --repetition_penalty 1
!pwd

!ls ../input/