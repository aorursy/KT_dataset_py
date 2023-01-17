# !git clone https://github.com/zihangdai/xlnet.git
!wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
!unzip cased_L-24_H-1024_A-16.zip
# !wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# !tar zxvf aclImdb_v1.tar.gz
!mkdir gs_root
!ls
!ls ./xlnet_cased_L-24_H-1024_A-16
# !python ./xlnet/run_classifier.py \

#   --use_tpu=False \

#   --do_train=True \

#   --do_eval=True \

#   --eval_all_ckpt=True \

#   --task_name=imdb \

#   --data_dir=./aclImdb \

#   --output_dir=./gs_root/proc_data/imdb \

#   --model_dir=./gs_root/exp/imdb \

#   --uncased=False \

#   --spiece_model_file=./xlnet_cased_L-24_H-1024_A-16/spiece.model \

#   --model_config_path=./xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \

#   --init_checkpoint=./xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \

#   --max_seq_length=512 \

#   --train_batch_size=32 \

#   --eval_batch_size=8 \

#   --num_hosts=1 \

#   --num_core_per_host=8 \

#   --learning_rate=2e-5 \

#   --train_steps=4000 \

#   --warmup_steps=500 \

#   --save_steps=500 \

#   --iterations=500