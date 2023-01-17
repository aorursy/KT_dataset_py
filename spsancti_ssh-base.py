import random, string, urllib.request, json, getpass



#Generate root password

password = 'qwertyuiop'
#Setup sshd

! apt-get install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen > /dev/null
#Set root password

! echo root:$password | chpasswd

! mkdir -p /var/run/sshd

! echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

! echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

! echo "LD_LIBRARY_PATH=/usr/lib64-nvidia" >> /root/.bashrc

! echo "export LD_LIBRARY_PATH" >> /root/.bashrc

! echo "Port 12345" >> /etc/ssh/sshd_config
#Run sshd

get_ipython().system_raw('/usr/sbin/sshd -D &')
get_ipython().system_raw('ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -R kaggle:12345:localhost:12345 serveo.net &')
import time

time.sleep(3600 * 8.9)