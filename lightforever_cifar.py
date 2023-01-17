! cp -a /kaggle/input/catalyst/catalyst/install.sh /tmp/install.sh && chmod 777 /tmp/install.sh && /tmp/install.sh /kaggle/input/catalyst/catalyst
cd /tmp/catalyst/examples
! catalyst-dl run --config cifar_simple/config.yml