import selectors

import subprocess

import sys



# runner (gdrive setting the same time) -> rvs.sh (setup reverse connection) ->

# setup pseudo tty

with open("runner.sh", "w") as f:

    f.write(

r"""#!/bin/bash -x

export PS4='Line ${LINENO}: '  # for debug

NC=ncat



USER=$1

shift

REPO=$1

shift

BRANCH=$1

shift

PHASE=$1

shift

PARAMS=$@



SERVER=vtool.duckdns.org

PORT=23454

CHECK_PORT=$(( PORT + 1 ))



apt install pv nmap screen time tmux netcat psmisc -y



# tmux new-session -d -s mySession -n myWindow

# tmux send-keys -t mySession:myWindow "echo debug" Enter

# tmux ls

pip install pysnooper  # for debug rvs

screen -d -m bash ./rvs.sh



pip install pydicom parse pytest-logger python_logging_rabbitmq &

# pip install parse  # should move local codes out

# pip install pytest-logger pysnooper python_logging_rabbitmq  # for debugging



cat > .tmux.conf <<EOF

# unbind-key C-b

# set -g prefix C-a

set -g prefix2 \

# bind-key C-o send-prefix

bind-key ` send `

bind-key C new-window \; command-prompt -p "Name for this new window: " "rename-window '%%'"

# set the default TERM

set -g default-terminal screen

set -g allow-rename off



# update the TERM variable of terminal emulator when creating a new session or attaching a existing session

set -g update-environment 'DISPLAY SSH_ASKPASS SSH_AGENT_PID SSH_CONNECTION WINDOWID XAUTHORITY TERM'

# determine if we should enable 256-colour support

if "[[ ${TERM} =~ 256color || ${TERM} == fbterm ]]" 'set -g default-terminal screen-256color'



# makes sure that if I try to attach and no sessions are alive, one is created.

# new-session -n $HOST



# 0 is too far from ` ;)

set -g base-index 1



# Automatically set window title

set-window-option -g automatic-rename off

set-option -g set-titles on



set -g status-keys vi

set -g history-limit 10000



setw -g mode-keys vi

setw -g mouse on

setw -g monitor-activity on



bind-key v split-window -h

bind-key s split-window -v



bind-key o display-panes

# bind-key h select-pane -L #resize-pane -D 5

# bind-key j select-pane -D #resize-pane -U 5

# bind-key k select-pane -U #resize-pane -L 5

# bind-key l select-pane -R #resize-pane -R 5



bind-key M-j resize-pane -D

bind-key M-k resize-pane -U

bind-key M-h resize-pane -L

bind-key M-l resize-pane -R



# Vim style pane selection

bind h select-pane -L

bind j select-pane -D

bind k select-pane -U

bind l select-pane -R



# Use Alt-vim keys without prefix key to switch panes

bind -n M-h select-pane -L

bind -n M-j select-pane -D

bind -n M-k select-pane -U

bind -n M-l select-pane -R





# Use Alt-arrow keys without prefix key to switch panes

# bind -n M-Left select-pane -L

# bind -n M-Right select-pane -R

# bind -n M-Up select-pane -U

# bind -n M-Down select-pane -D



# Alt-[".",","] to switch windows

bind -n M-. next-window

bind -n M-, previous-window



# No delay for escape key press

set -sg escape-time 0



# Reload tmux config

bind r source-file ~/.tmux.conf



# set -g status-utf8 on

set -g status-justify left

set -g status-interval 30



# window status

setw -g window-status-format " #F#I:#W#F "

setw -g window-status-current-format " #F#I:#W#F "

setw -g window-status-format "#[fg=magenta]#[bg=black] #I #[bg=cyan]#[fg=colour8] #W "

setw -g window-status-current-format "#[bg=brightmagenta]#[fg=colour8] #I #[fg=colour8]#[bg=colour14] #W "

set -g status-left-length 30

set -g status-left 'P#P #[fg=green][#W] in (#S)#[fg=gray] as #(whoami) '

# set -g status-right '#[fg=yellow]#(cut -d " " -f 1-3 /proc/loadavg)#[default] #[fg=white]%H:%M#[default]'

set -g status-right-length 60



# loud or quiet?

set-option -g visual-activity off

set-option -g visual-bell off

set-option -g visual-silence off

set-window-option -g monitor-activity off

set-option -g bell-action none



unbind [

bind [ copy-mode

unbind p

bind p paste-buffer

bind-key -Tcopy-mode-vi 'C-v' send -X begin-selection \; send -X rectangle-toggle

bind-key -Tcopy-mode-vi 'v' send -X begin-selection

bind-key -Tcopy-mode-vi 'y' send -X copy-selection-and-cancel





# The modes {

setw -g clock-mode-colour colour135

setw -g mode-style bg=colour238,fg=colour196,bold



# }

# The panes {



# highlight selected pane, tab

# set -g window-style 'fg=colour247,bg=colour236'

# set -g window-active-style 'fg=colour250,bg=black'

# set -g pane-border-style fg=colour238,bg=colour235

set -g pane-border-style bg=colour236,fg=colour51

set -g pane-active-border-style bg=colour236,fg=colour51



# }

# The statusbar {



set -g status-position bottom

set -g status-style bg=colour234,fg=colour137,dim

set -g status-right '#[fg=colour233,bg=colour241,bold] %d/%m #[fg=colour233,bg=colour245,bold] %H:%M:%S '

set -g status-right-length 50



setw -g window-status-current-style bg=colour238,fg=colour81,bold

setw -g window-status-current-format ' #I#[fg=colour250]:#[fg=colour255]#W#[fg=colour50]#F '



setw -g window-status-style bg=colour235,fg=colour138,none

setw -g window-status-format ' #I#[fg=colour237]:#[fg=colour250]#W#[fg=colour244]#F '



setw -g window-status-bell-style bg=colour1,fg=colour255,bold



# }

# The messages {



set -g message-style bg=colour166,fg=colour232,bold



# }



# List of plugins

set -g @plugin 'tmux-plugins/tpm'

set -g @plugin 'tmux-plugins/tmux-sensible'

set -g @plugin 'tmux-plugins/tmux-resurrect'

set -g @plugin 'tmux-plugins/tmux-continuum'

set -g @plugin 'tmux-plugins/tmux-yank'

set -g @plugin 'tmux-plugins/tmux-sensible'

set -g @continuum-restore 'on'



# Other examples:

# set -g @plugin 'github_username/plugin_name'

# set -g @plugin 'git@github.com/user/plugin'

# set -g @plugin 'git@bitbucket.com/user/plugin'



# Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)

run -b '~/.tmux/plugins/tpm/tpm'

bind | split-window -h

bind - split-window -v

bind-key P command-prompt -p 'save history to filename:' -I '~/tmux.history' 'capture-pane -S -32768 ; save-buffer %1 ; delete-buffer'

bind-key W command-prompt -p "Switch to pane with pid:" "run-shell 'pane=\$(ps eww %% | sed \"1d; s/^.*TMUX_PANE=//;s/ .*//\"); [[ -z \$pane ]] && tmux display-message \"could not find pid\" || tmux switch-client -t \$pane'"

EOF



cat > ENVS <<EOF

# CUDNN_VERSION=7.6.5.32

# LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:

# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

LESSCLOSE=/usr/bin/lesspipe %s %s

LANG=en_US.UTF-8

# HOSTNAME=8bff88b8a353

OLDPWD=/

CLOUDSDK_CONFIG=/content/.config

GOOGLE_APPLICATION_CREDENTIALS=/content/adc.json

NVIDIA_VISIBLE_DEVICES=all

DATALAB_SETTINGS_OVERRIDES={kernelManagerProxyPort:6000,kernelManagerProxyHost:172.28.0.3,jupyterArgs:[--ip="172.28.0.2"]}

ENV=/root/.bashrc

PAGER=cat

NCCL_VERSION=2.4.8

TF_FORCE_GPU_ALLOW_GROWTH=true

JPY_PARENT_PID=18

NO_GCE_CHECK=True

PWD=/content

# HOME=/root

LAST_FORCED_REBUILD=20200316

CLICOLOR=1

DEBIAN_FRONTEND=noninteractive

LIBRARY_PATH=/usr/local/cuda/lib64/stubs

GCE_METADATA_TIMEOUT=0

GLIBCPP_FORCE_NEW=1

TBE_CREDS_ADDR=172.28.0.1:8008

SHELL=bash

TERM=xterm-256color

GCS_READ_CACHE_BLOCK_SIZE_MB=16

PYTHONWARNINGS=ignore:::pip._internal.cli.base_command

MPLBACKEND=module://ipykernel.pylab.backend_inline

# CUDA_PKG_VERSION=10-1=10.1.243-1

# CUDA_VERSION=10.1.243

# NVIDIA_DRIVER_CAPABILITIES=compute,utility

SHLVL=3

# PYTHONPATH=/env/python

# NVIDIA_REQUIRE_CUDA=cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411

# COLAB_GPU=0

# GLIBCXX_FORCE_NEW=1

# PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin

# PS1=\[\033[01;32m\]\u@\h \[\033[01;34m\]\w \[\033[31m\]`git branch 2> /dev/null | grep -e ^* | sed -E  s/^\\\\\*\ \(.+\)$/\(\\\\\1\)\ /`\[\033[35m\]$\[\033[00m\]

PS4=+

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

LESSOPEN=| /usr/bin/lesspipe %s

GIT_PAGER=cat

_=/usr/bin/env

EOF





SRC_WORK_FOLDER=/kaggle/working

[ -d ${SRC_WORK_FOLDER} ] || mkdir -p ${SRC_WORK_FOLDER}

cd ${SRC_WORK_FOLDER}

(test -d ${REPO} || git clone --single-branch --branch ${BRANCH} --depth=1 \

https://github.com/${USER}/${REPO}.git ${REPO} && pushd ${REPO} && \

 find . -maxdepth 1 -name ".??*" -o -name "??*" | xargs -I{} mv {} $OLDPWD && popd) \

 && {

     if [x"${PHASE}" != x"dev"]; then

         python main.py $PARAMS;

     else

         # just two, incase another one goes down

         PS4='Line ${LINENO}: ' bash -x ./rvs.sh | $NC $SERVER $CHECK_PORT;

     fi

    }

# GRAMMAR: NAME () COMPOUND-COMMAND [ REDIRECTIONS ]

# while true; do sleep 60; done  # just wait

"""

    )

with open("rvs.sh", "w") as f:

    f.write(

r"""#!/bin/bash -x

export PS4='Line ${LINENO}: ' # for debug

NC=ncat



# https://stackoverflow.com/questions/57877451/retrieving-output-and-exit-code-of-a-coprocess

# coproc { sleep 30 && echo "Output" && exit 3; }

# Saving the coprocess's PID for later, as COPROC_PID apparently unsets when its finished

# COPROC_PID_backup=$COPROC_PID

#

# Retrieving the coprocess's output

# output=$(cat <&$COPROC)

#

# Retrieving the coprocess's exit code

# wait $COPROC_PID_backup

#

# Echoing out the results

# echo $?

# echo $output



waitfile() {

  while [ ! -f $1 ]; do

    sleep 1

  done

}



echo BASH NOW: $BASHPID



PID_FILE_PATH=/tmp/nc.pid

EXIT_FILE_PATH=/tmp/rvs_exit.$BASHPID.pid



test -f $EXIT_FILE_PATH && rm $EXIT_FILE_PATH



SERVER=vtool.duckdns.org

PORT=23454

CHECK_PORT=$((PORT + 1))



check_exit_status() {

  if [ -f $EXIT_FILE_PATH -a x$(cat $EXIT_FILE_PATH) = x0 ]; then

    return 0

  fi

  return 1 # not ok

}

connect_to_server() {

  cat rpt

  echo "#" $(date) started connection

  $NC -w $1 $SERVER $PORT

} 2>&1

connect_setup() {

  check_exit_status && return 0



  # The standard output of COMMAND is connected via a pipe to a file

  # descriptor in the executing shell, and that file descriptor is assigned

  # to 'NAME'[0].  The standard input of COMMAND is connected via a pipe to

  # a file descriptor in the executing shell, and that file descriptor is

  # assigned to 'NAME'[1].  This pipe is established before any redirections

  # specified by the command (*note Redirections::).

  PID_FILE_PATH=$PID_FILE_PATH.$BASHPID

  (

    coproc connect_to_server $1

    COPROC_PID_backup=$COPROC_PID

    echo $COPROC_PID_backup $PID_FILE_PATH # debug

    echo $COPROC_PID_backup > $PID_FILE_PATH

    # exec -l bash <&${COPROC[0]} >&${COPROC[1]} 2>&1;

    # COPROC[0] is the output of nc

    exec -l python setup_pty log_master log_log <&${COPROC[0]} >&${COPROC[1]} 2>&1

  )

  RSPID=$!

  wait $RSPID # what about connection loss? need to check heatbeat

  RSRET=$?

  if [ x"$RSRET" = x"0" ]; then  # TODO fix, named pipe, return always 120?

    echo $RSRET > $EXIT_FILE_PATH

    return $RSRET

  fi

  # else part below



  waitfile $PID_FILE_PATH &&

    tail --pid=$(cat $PID_FILE_PATH) -f /dev/null &&

    rm $PID_FILE_PATH



  pgrep $RSPID && kill $RSPID

  # just recursively, sleep in case...

  sleep 5 && [ ! $RSRET -eq 120 ] && connect_again

  # exit, will cause rvs script exit, beside, RSRET not 0, mean connection loss thing

  echo $RSRET > $EXIT_FILE_PATH && return $RSRET

}



connect_again() {

  # pkill -f "nc.*$PORT"  # no need now, our listen server can accept multiple

  # connection now

  connect_setup $1 & # just put connection to background

}



WAIT_LIMIT=128

INIT_WAIT=8

port_connect_status=0

wait_time=$INIT_WAIT



floatToInt() {

  parsed=$(printf "%.0f" "$@")

  [ ! $? -eq 0 ] && parsed=0

  echo $parsed

} 2> /dev/null



while true; do

  check_exit_status && exit 0

  # if find that server cannot be connected, we try to restart our reverse connect again

  nc_time=$($(which time) -f "%e" $NC -zw $wait_time $SERVER $CHECK_PORT 2>&1 > /dev/null)

  nc_ret=$?

  nc_time=$(echo $nc_time | awk '{print $NF}')

  nc_time=$(floatToInt $nc_time)

  if [ ${nc_ret} -eq 0 ]; then

    # recover connection, need to connect_again too. For 1st time, will try to connect

    # no connection last time, have connction now

    if [ $port_connect_status -eq 0 ]; then

      echo "recover connection, reset wait_time and try to reconnect"

      wait_time=$INIT_WAIT

      # previous connection is lost, we wait for longer to setup connection

      check_exit_status || wait_time=15

      connect_again $wait_time

    else

      wait_time=$((wait_time + wait_time)) # double wait, network fine

      if [ $wait_time -gt ${WAIT_LIMIT} ]; then wait_time=${WAIT_LIMIT}; fi

    fi

    port_connect_status=1

  else

    if [ $port_connect_status -eq 1 ]; then

      echo "found connection loss, reset wait_time and try to reconnect"

      wait_time=$INIT_WAIT

      check_exit_status || wait_time=15 # previous connection is lost

      connect_again $wait_time

    else # no connection all the time? we still try to connect...

      wait_time=$((wait_time + wait_time))

      if [ $wait_time -gt ${WAIT_LIMIT} ]; then wait_time=${WAIT_LIMIT}; fi

      connect_again $wait_time

    fi

    port_connect_status=0

  fi

  sleep $((wait_time - nc_time)) # check every XX seconds

done



# https://medium.com/@6c2e6e2e/spawning-interactive-reverse-shells-with-tty-a7e50c44940e

# In reverse shell

# $ python -c 'import pty; pty.spawn("/bin/bash")'

# Ctrl-Z

#

# In Attacker console

# $ stty raw -echo

# $ fg

#

# In reverse shell

# $ reset

# $ export SHELL=bash

# $ export TERM=xterm-256color

# $ stty rows <num> columns <cols>

while true; do sleep 60; done

"""

    )

with open("setup_pty", "w") as f:

    f.write(

r"""import argparse

import os

import pty

import sys

import time



parser = argparse.ArgumentParser()

parser.add_argument('-a', dest='append', action='store_true')

parser.add_argument('-p', dest='use_python', action='store_true')

parser.add_argument('filename', nargs='?', default='typescript')

parser.add_argument('logfilename', nargs='?', default='typescript')

options = parser.parse_args()



shell = sys.executable if options.use_python else os.environ.get('SHELL', 'sh')

filename = options.filename

logfilename = options.logfilename

mode = 'ab' if options.append else 'wb'



def main():

    with open(filename, mode) as script:

        def read(fd):

            data = os.read(fd, 1024)

            script.write(data)

            return data



        with open(logfilename, mode) as logscript:

            def logread(fd):

                data = os.read(fd, 1024)

                logscript.write(data)

                return data



            print('Script started, file is', filename)

            script.write(('Script started on %s\n' % time.asctime()).encode())



            pty.spawn(shell, read, logread)



            script.write(('Script done on %s\n' % time.asctime()).encode())

            print('Script done, file is', filename)



if __name__ == "__main__":

     main()

"""

    )

with open("rpt", "w") as f:

    f.write(

r"""#!/bin/bash

reset

export SHELL=bash

export TERM=xterm-256color

stty intr ^\k susp ^\x eof ^\f -echo rows 29 columns 59 opost

# https://unix.stackexchange.com/questions/343088/what-is-the-equivalent-of-stty-echo-for-zsh

unsetopt ZLE # for zsh

# for ourside stty raw isig -echo icrnl time 3 echoprt opost eof ^\p



color_my_prompt () {

    local __user_and_host="\[\033[01;32m\]\u@\h"

    local __cur_location="\[\033[01;34m\]\w"

    local __git_branch_color="\[\033[31m\]"

    # local __git_branch="\`ruby -e \"print (%x{git branch 2> /dev/null}.grep(/^\*/).first || '').gsub(/^\* (.+)$/, '(\1) ')\"\`"

    local __git_branch='`git branch 2> /dev/null | grep -e ^* | sed -E  s/^\\\\\*\ \(.+\)$/\(\\\\\1\)\ /`'

    local __prompt_tail="\[\033[35m\]$"

    local __last_color="\[\033[00m\]"

    export PS1="$__user_and_host $__cur_location $__git_branch_color$__git_branch$__prompt_tail$__last_color "

}

color_my_prompt



# CUDNN_VERSION=7.6.5.32

# LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:

# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

LESSCLOSE=/usr/bin/lesspipe %s %s

LANG=en_US.UTF-8

# HOSTNAME=8bff88b8a353

OLDPWD=/

CLOUDSDK_CONFIG=/content/.config

GOOGLE_APPLICATION_CREDENTIALS=/content/adc.json

NVIDIA_VISIBLE_DEVICES=all

DATALAB_SETTINGS_OVERRIDES={kernelManagerProxyPort:6000,kernelManagerProxyHost:172.28.0.3,jupyterArgs:[--ip="172.28.0.2"]}

ENV=/root/.bashrc

PAGER=cat

NCCL_VERSION=2.4.8

TF_FORCE_GPU_ALLOW_GROWTH=true

JPY_PARENT_PID=18

NO_GCE_CHECK=True

PWD=/content

# HOME=/root

LAST_FORCED_REBUILD=20200316

CLICOLOR=1

DEBIAN_FRONTEND=noninteractive

LIBRARY_PATH=/usr/local/cuda/lib64/stubs

GCE_METADATA_TIMEOUT=0

GLIBCPP_FORCE_NEW=1

TBE_CREDS_ADDR=172.28.0.1:8008

SHELL=bash

TERM=xterm-256color

GCS_READ_CACHE_BLOCK_SIZE_MB=16

PYTHONWARNINGS=ignore:::pip._internal.cli.base_command

MPLBACKEND=module://ipykernel.pylab.backend_inline

# CUDA_PKG_VERSION=10-1=10.1.243-1

# CUDA_VERSION=10.1.243

# NVIDIA_DRIVER_CAPABILITIES=compute,utility

SHLVL=3

# PYTHONPATH=/env/python

# NVIDIA_REQUIRE_CUDA=cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411

# COLAB_GPU=0

# GLIBCXX_FORCE_NEW=1

# PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin

# PS1=\[\033[01;32m\]\u@\h \[\033[01;34m\]\w \[\033[31m\]`git branch 2> /dev/null | grep -e ^* | sed -E  s/^\\\\\*\ \(.+\)$/\(\\\\\1\)\ /`\[\033[35m\]$\[\033[00m\]

PS4=+

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

LESSOPEN=| /usr/bin/lesspipe %s

GIT_PAGER=cat

_=/usr/bin/env

"""

    )

with open("gdrive_setup", "w") as f:

    f.write(

r"""#!/bin/bash

wget https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64

chmod +x gdrive-linux-x64

cp gdrive-linux-x64 /bin/gdrive



mkdir ~/.gdrive



# auth file

cat > ~/.gdrive/a.json << EOF

{

  "type": "service_account",

  "project_id": "go-2-learn",

  "private_key_id": "00c8bf796e900c9afe68129c9fbdef6f42084bef",

  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC8XTKTpSpxiiu+\nB29mfkVWVwjUZhoYBC9td8QtjZYRGaa0HNPSPYQkKY69GGGGwbzocuuhhFFW9P8g\nCFbjKIfLKTgJTBJ0PZy78LmZL+YjUo7N/x3MMRQKPzY4JSMQVVTVXhHBD2IxtFJ5\nKD6tojmuLblaNEUn2kwWvMBMZnTmtHZkNVsnRWhqNYxoJePGJOygDbWmUGyMLDmY\nihdysNdvNYGyI9pVwQCMwEWesuykA438cehj9v0p3YunghDa2EBH7GKAAk4aoie3\ndoNjmGeOV/DXb01heLBxlumhQX3qGt+cpVSv7wFQRq/z+CMsK1rkbJmE+WZg8EAE\nOJ3HxPJ1AgMBAAECggEAFfNxHBj4rpfrgQ8HbGpKqkUk7PD5GX4OCN5sKOLXGicN\nxkTrFRUWJnYGrFgAWt6OT92/QpNTkflQbJXhikIEO9NnNFjTzbgLC9vXGoL6eXil\ngQa58jxwmWvEZcaYz3kiPxCMq8hJ09ZacMQVNHwzPJkXgJZBeON3pS6vOjgLvNbC\ncQLKPl9NU4GucXECtdCdsLxhfAxkgbPrHh2DBkIX5jw+qB1DpkxfnT/icuNW6yWo\nb8snh/zU58iHNGlXwdWoTIajrRPBUEbAO/KBbkhz3H6o/GdMDEx4zg/lhDmvEDFn\nuCOtoXHZxYgkIbUEZGRMmbbrcGN80ppaGI9uLOaWAQKBgQDdigypRFzs7bqlN1eJ\nTEn45WqmLrZcNFs1veqCAMOdD6QjOM18zigA8QqDP8d8dD1fFwQEyEy8j8303uTo\ntKQjvPudKgYn0LfUqSM1gI5oIjgpauTbfxK8Vzvly6LyobnM2C/ZawUI/LZ81VvQ\nZIbYjK5Cmb+rm3brykDWlerVKwKBgQDZqhKvs8ALqQb+gUV0WQdZOcNX8cYF3ADa\nfV+Lqj8quEcxopdmtNNv6VA6O6/B74gHkAj+2s3Azf2RKmIcsNZ0ETurSh2iOHul\n6kz01R8FVbF47iPnkurJvRpb1JLQUJZCtzSSsftK2ZPPCQe9MNN+Hms3kVb2dJuV\nvN3PLwLG3wKBgC0DU7dA0LDDTN0s9XhMK+uKkbTaYOszKCUvRWrMxPIwr2UIsZfe\nO3qVf1FTsDC1XZLolkRyfkUB4xMSBujRa1hnmahBVabZXcCz7Rd923GFImwn8AA5\nPZFPGDiEu8MY4Suh8Xb3q7o7vsh2gYVCJ7PwQaf+nVc861jVa38uTtypAoGAJ5lk\naujF0JlAt36nNyKXTqlOm6pVv20mDpnujwc7FLeP5DzTVJEjQmHtAZsoP50nX1Da\nAhumgSQ4tHdEgDm/2j/kXiZOu9uQyz+UHprDWQIdFoYkrBWzd15a9Ef5KcLvg1W3\nT9TnhdeNp4XaDZZbc79u/B4J9y6Bu70vkWjZFXsCgYEAzD/vhTXZQeBDM9oMOMp2\nkukN3jHwG2cAFb5127ygJHZns2071AwoeQYnpsnpAC4z7C7sjoeYWfktno5oFbnq\nGyquWXdHrC/jELYKcgJzZgncoXjilI8EODvq7a6GVGp4Rlz1mGQW7BQnHcPM+Jri\n/hjKqXujhl6U29XgaDkDEUk=\n-----END PRIVATE KEY-----\n",

  "client_email": "drive-322@go-2-learn.iam.gserviceaccount.com",

  "client_id": "112374369447770992406",

  "auth_uri": "https://accounts.google.com/o/oauth2/auth",

  "token_uri": "https://oauth2.googleapis.com/token",

  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",

  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/drive-322%40go-2-learn.iam.gserviceaccount.com"

}

EOF



gdrive --service-account a.json list  # just test



SRC_WORK_FOLDER=/kaggle/input

[ -d ${SRC_WORK_FOLDER} ] || {

    mkdir -p ${SRC_WORK_FOLDER}

    cd ${SRC_WORK_FOLDER}

    gdrive --service-account a.json download -r 1CHDWIN0M6PD4SQyplbWefBCzNzdPVd-m

    tar xf siim-train-test.tar.gz -C /kaggle/input

}

# cat > tgz_files.sh << EOF

# #!/bin/bash

# tgzfile () {

#   tar cf - $1 -P | pv -s $(du -sb $1 | awk '{print $1}') | gzip > /home/$1.tar.gz

# }

# cd /kaggle/input

# find . -maxdepth 1 -type d -name "??*" | while read -r line; do

#     echo $line

#     tgzfile $line

# done

# EOF

"""

    )

with open("entry.sh", "w") as f:

    f.write(

r"""#!/bin/bash

# PS4='Line ${LINENO}: ' bash -x gdrive_setup >>loggdrive &

# PS4='Line ${LINENO}: ' bash -x runner.sh pennz PneumothoraxSegmentation dev dev "amqp://drdsfaew:QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI@termite.rmq.cloudamqp.com/drdsfaew" "384" "19999" "intercept-resnet" >>logrunner

"""

    )





p = subprocess.Popen(

'bash -x entry.sh',

#capture_output=True,

stdout=subprocess.PIPE, stderr=subprocess.PIPE,

shell=True)



sel = selectors.DefaultSelector()

sel.register(p.stdout, selectors.EVENT_READ)

sel.register(p.stderr, selectors.EVENT_READ)



while True:

   for key, _ in sel.select():

       data = key.fileobj.read1().decode()

       if not data:

           exit()

       if key.fileobj is p.stdout:

           print(data, end="")

       else:

           print(data, end="", file=sys.stderr)

# URL:

# https://stackoverflow.com/questions/31833897/python-read-from-subprocess-stdout-and-stderr-separately-while-pr

# eserving-order

# Title: Python read from subprocess stdout and stderr separately while preserving order - Stack Overflow

#

# Size: 110088

# Codepage: Unicode UTF-8

# SSL Cipher: 128-bit TLSv1.2 ECDHE-RSA-AES128-GCM-SHA256

# Encoding: gzip

# Date: Mon, 04 May 2020 23:38:11 GMT

# Last modified: Mon, 04 May 2020 23:38:11 GMT

# Time since loading: 13:08

# Last visit time: Tue May  5 07:41:07 2020

#

# Link: https://stackoverflow.com/a/56918582

# Link title: short permalink to this answer



# just a test problem

# import sys

# from time import sleep

#

# for i in range(10):

#     print(f" x{i} ", file=sys.stderr, end="")

#     sleep(0.1)

#     print(f" y{i} ", end="")

#     sleep(0.1)
#%run /opt/conda/bin/pytest --pdb -s -k "test_pytorch"