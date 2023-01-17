!git clone https://github.com/Luvata/CS224N-2019.git
%cd CS224N-2019/
# remove this folder or you'll get error when commit

!rm -rf .git/

!rm -rf note/
%cd Assignment/
!rm -rf a1 a2 a3 a4 # remove redundant directory
%cd a5-v1.2/
!sh run.sh vocab # make vocab file first
!sh run.sh train # train and wait :D
!sh run.sh test # will show BLEU score