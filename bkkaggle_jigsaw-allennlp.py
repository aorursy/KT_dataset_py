!pip install python-dateutil --upgrade

!pip install awscli --upgrade

!pip install greenlet --ignore-installed --upgrade

!mkdir packages
%%writefile packages/requirements.txt

python-dateutil

awscli

greenlet

allennlp
!pip download -r packages/requirements.txt -d packages
!tar -zcf packages.tar.gz packages
!rm -rf packages















