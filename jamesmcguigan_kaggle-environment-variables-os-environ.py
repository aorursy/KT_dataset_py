import os

print(f"os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost') == '{os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost')}'")



if os.environ.get('KAGGLE_KERNEL_RUN_TYPE',''):

    print("os.environ.get('KAGGLE_KERNEL_RUN_TYPE','')                  | We are running a Kaggle Notebook/Script - Could be Interactive or Batch Mode")  

    

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Interactive':

    print("os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Interactive' | We are running a Kaggle Notebook/Script - Interactive Mode")



if os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Batch':

    print("os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Batch'       | We are running a Kaggle Notebook/Script - Batch Mode")



if os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == '':

    print("os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == ''            | We are running code on Localhost")    
# os.environ['KAGGLE_URL_BASE'] == 'https://www.kaggle.com'

print("os.environ.get('KAGGLE_URL_BASE') == ", os.environ.get('KAGGLE_URL_BASE'))

if 'kaggle' in os.environ.get('KAGGLE_URL_BASE','localhost'):

    print("We are running on a Kaggle Server") 
print("os.environ.get('PWD') == ", os.environ.get('PWD'))

if os.environ.get('PWD') == '/kaggle/working':

    print("We are running on a Kaggle Server")
# os.environ['KAGGLE_URL_BASE'] == 'module://ipykernel.pylab.backend_inline'  

print("os.environ.get('MPLBACKEND') == ", os.environ.get('MPLBACKEND'))

if os.environ.get('MPLBACKEND') == 'agg':

    print("We are editing a Kaggle Notebook")

if os.environ.get('MPLBACKEND') == 'module://ipykernel.pylab.backend_inline':

    print('We are editing on Localhost')
print("os.environ.get('USERNAME') == ", os.environ.get('USERNAME'))

if os.environ.get('USERNAME'):

    print('We are editing in IntelliJ')
!find /kaggle/
import os

os_environ = {}



{ k: (v if 'TOKEN' not in k else "#") for k,v in sorted(os.environ.items()) }
# When running on Kaggle in batch mode after hitting the commit button

os_environ['kaggle_commit_batch'] = {

    '_': '/opt/conda/bin/jupyter',

    'CLICOLOR': '1',

    'GIT_PAGER': 'cat',

    'HOME': '/root',

    'HOSTNAME': '1d07050df1ff',

    'JPY_PARENT_PID': '8',

    'KAGGLE_DATA_PROXY_PROJECT': 'kaggle-161607',

    'KAGGLE_DATA_PROXY_TOKEN': '#',

    'KAGGLE_DATA_PROXY_URL': 'https://dp.kaggle.net',

    'KAGGLE_KERNEL_INTEGRATIONS': '',

    'KAGGLE_KERNEL_RUN_TYPE': 'Batch',

    'KAGGLE_URL_BASE': 'https://www.kaggle.com',

    'KAGGLE_USER_SECRETS_TOKEN': '#',

    'LANG': 'C.UTF-8',

    'LC_ALL': 'C.UTF-8',

    'LD_LIBRARY_PATH': '/opt/conda/lib',

    'MKL_THREADING_LAYER': 'GNU',

    'MPLBACKEND': 'agg',

    'OLDPWD': '/kaggle/working',

    'PAGER': 'cat',

    'PATH': '/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',

    'PROJ_LIB': '/opt/conda/share/proj',

    'PWD': '/kaggle/working',

    'PYTHONPATH': '/kaggle/lib/kagglegym:/kaggle/lib',

    'PYTHONUSERBASE': '/root/.local',

    'SHLVL': '1',

    'TERM': 'xterm-color',

    'TESSERACT_PATH': '/usr/bin/tesseract',

}



# When editing in a Kaggle Interative Notebook

os_environ['kaggle_interactive'] = {

    '_': '/opt/conda/bin/jupyter',

    'CLICOLOR': '1',

    'GIT_PAGER': 'cat',

    'HOME': '/root',

    'HOSTNAME': 'ea8b88d7cf35',

    'JPY_PARENT_PID': '8',

    'KAGGLE_DATA_PROXY_PROJECT': 'kaggle-161607',

    'KAGGLE_DATA_PROXY_TOKEN': '#',

    'KAGGLE_DATA_PROXY_URL': 'https://dp.kaggle.net',

    'KAGGLE_KERNEL_INTEGRATIONS': '',

    'KAGGLE_KERNEL_RUN_TYPE': 'Interactive',

    'KAGGLE_URL_BASE': 'https://www.kaggle.com',

    'KAGGLE_USER_SECRETS_TOKEN': '#',

    'LANG': 'C.UTF-8',

    'LC_ALL': 'C.UTF-8',

    'LD_LIBRARY_PATH': '/opt/conda/lib',

    'MKL_THREADING_LAYER': 'GNU',

    'MPLBACKEND': 'agg',

    'PAGER': 'cat',

    'PATH': '/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',

    'PROJ_LIB': '/opt/conda/share/proj',

    'PWD': '/kaggle/working',

    'PYTHONPATH': '/kaggle/lib/kagglegym:/kaggle/lib',

    'PYTHONUSERBASE': '/root/.local',

    'SHLVL': '1',

    'TERM': 'xterm-color',

    'TESSERACT_PATH': '/usr/bin/tesseract',

}





# When running localhost Jupyter Lab from a Terminal and editing in a Webbrowser 

os_environ['localhost_jupyter_lab'] = {

    'BAMF_DESKTOP_FILE_HINT': '/var/lib/snapd/desktop/applications/intellij-idea-ultimate_intellij-idea-ultimate.desktop',

    'CLICOLOR': '1',

    'CLUTTER_IM_MODULE': 'xim',

    'COMP_CONFIGURE_HINTS': '1',

    'COMP_CVS_REMOTE': '1',

    'COMP_TAR_INTERNAL_PATHS': '1',

    'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus',

    'DESKTOP_SESSION': 'ubuntu',

    'DISPLAY': ':1',

    'EDITOR': 'vim',

    'GDK_BACKEND': 'x11',

    'GDMSESSION': 'ubuntu',

    'GIO_LAUNCHED_DESKTOP_FILE': '/var/lib/snapd/desktop/applications/intellij-idea-ultimate_intellij-idea-ultimate.desktop',

    'GIO_LAUNCHED_DESKTOP_FILE_PID': '10875',

    'GIT_PAGER': 'cat',

    'GJS_DEBUG_OUTPUT': 'stderr',

    'GJS_DEBUG_TOPICS': 'JS ERROR;JS LOG',

    'GNOME_DESKTOP_SESSION_ID': 'this-is-deprecated',

    'GNOME_SHELL_SESSION_MODE': 'ubuntu',

    'GPG_AGENT_INFO': '/run/user/1000/gnupg/S.gpg-agent:0:1',

    'GTK_IM_MODULE': 'ibus',

    'GTK_MODULES': '',

    'HISTCONTROL': 'ignoreboth',

    'HOME': '/home/jamie',

    'HOMEBREW_GITHUB_API_TOKEN': '#',

    'IM_CONFIG_PHASE': '2',

    'JAVA_HOME': '',

    'JPY_PARENT_PID': '2737',

    'KERNEL_LAUNCH_TIMEOUT': '40',

    'LANG': 'en_GB.UTF-8',

    'LANGUAGE': 'en_GB:en',

    'LD_LIBRARY_PATH': '/usr/local/cuda-10.1/targets/x86_64-linux/lib',

    'LESS': '-R',

    'LESSCLOSE': '/usr/bin/lesspipe %s %s',

    'LESSOPEN': '| /usr/bin/lesspipe %s',

    'LOGNAME': 'jamie',

    'MPLBACKEND': 'module://ipykernel.pylab.backend_inline',

    'NVM_BIN': '/home/jamie/.nvm/versions/node/v12.16.0/bin',

    'NVM_CD_FLAGS': '',

    'NVM_DIR': '/home/jamie/.nvm',

    'OLDPWD': '/home/jamie/code',

    'PAGER': 'cat',

    'PATH': './venv/bin:/home/jamie/code/kaggle-bengali-ai/venv/bin:/home/jamie/.nvm/versions/node/v12.16.0/bin:/usr/local/bin:/usr/local/sbin/:/usr/local/opt/coreutils/libexec/gnubin:/home/jamie/.local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:./.anaconda3/bin:./.local/bin:./node_modules/.bin:~/.config/yarn/global/node_modules/.bin/:~/.cabal/bin:~/.cargo/bin:~/.local/bin:~/.rvm/bin:~/Library/Haskell/bin:~/.git-scripts:~/Dropbox/Programming/command-line-scripts:/usr/local/cuda-10.1/bin',

    'PS1': "(venv)  \\n\\[\\e[32;1m\\][\\t]\\[\\e[35;1m\\]\\u\\[\\e[37;1m\\]@\\[\\e[35;1m\\]jamie-Blade-Pro\\[\\e[37;1m\\]:\\[\\e[33;1m\\]\\w$(__git_ps1 '\\[\\e[37;1m\\]|\\[\\e[31;1m\\]%s')\\[\\e[37;1m\\]\\$ \\[\\e[0m\\]\\n",

    'PWD': '/home/jamie/code/kaggle-bengali-ai',

    'PYTHONPATH': '/home/jamie/code/kaggle-bengali-ai',

    'PYTHONSTARTUP': './.pythonstartup.py',

    'QT4_IM_MODULE': 'xim',

    'QT_ACCESSIBILITY': '1',

    'QT_IM_MODULE': 'ibus',

    'SESSION_MANAGER': 'local/jamie-Blade-Pro:@/tmp/.ICE-unix/2275,unix/jamie-Blade-Pro:/tmp/.ICE-unix/2275',

    'SHELL': 'bash',

    'SHLVL': '1',

    'SNAP': '/snap/intellij-idea-ultimate/204',

    'SNAP_ARCH': 'amd64',

    'SNAP_COMMON': '/var/snap/intellij-idea-ultimate/common',

    'SNAP_CONTEXT': 'jv7sWxCBpTBHp5RC3XX8yqsmnQgpCRKfcG7QPK6k8Y5V',

    'SNAP_COOKIE': 'jv7sWxCBpTBHp5RC3XX8yqsmnQgpCRKfcG7QPK6k8Y5V',

    'SNAP_DATA': '/var/snap/intellij-idea-ultimate/204',

    'SNAP_INSTANCE_KEY': '',

    'SNAP_INSTANCE_NAME': 'intellij-idea-ultimate',

    'SNAP_LIBRARY_PATH': '/var/lib/snapd/lib/gl:/var/lib/snapd/lib/gl32:/var/lib/snapd/void',

    'SNAP_NAME': 'intellij-idea-ultimate',

    'SNAP_REEXEC': '',

    'SNAP_REVISION': '204',

    'SNAP_USER_COMMON': '/home/jamie/snap/intellij-idea-ultimate/common',

    'SNAP_USER_DATA': '/home/jamie/snap/intellij-idea-ultimate/204',

    'SNAP_VERSION': '2019.3.3',

    'SSH_AGENT_PID': '2386',

    'SSH_AUTH_SOCK': '/run/user/1000/keyring/ssh',

    'TERM': 'xterm-color',

    'TERMINAL_EMULATOR': 'JetBrains-JediTerm',

    'TEXTDOMAIN': 'im-config',

    'TEXTDOMAINDIR': '/usr/share/locale/',

    'USER': 'jamie',

    'VIRTUAL_ENV': '/home/jamie/code/kaggle-bengali-ai/venv',

    'WINDOWPATH': '2',

    'XAUTHORITY': '/run/user/1000/gdm/Xauthority',

    'XDG_CONFIG_DIRS': '/etc/xdg/xdg-ubuntu:/etc/xdg',

    'XDG_CURRENT_DESKTOP': 'ubuntu:GNOME',

    'XDG_DATA_DIRS': '/usr/share/ubuntu:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop',

    'XDG_MENU_PREFIX': 'gnome-',

    'XDG_RUNTIME_DIR': '/run/user/1000/snap.intellij-idea-ultimate',

    'XDG_SEAT': 'seat0',

    'XDG_SESSION_DESKTOP': 'ubuntu',

    'XDG_SESSION_ID': '2',

    'XDG_SESSION_TYPE': 'x11',

    'XDG_VTNR': '2',

    'XMODIFIERS': '@im=ibus',

    '_': './venv/bin/jupyter',

    'cabal_helper_libexecdir': '/home/jamie/.local/bin/'

}



# When using IntelliJ as an inline Jupyter IDE    

os_environ['localhost_intellij'] = {

    'BAMF_DESKTOP_FILE_HINT': '/var/lib/snapd/desktop/applications/intellij-idea-ultimate_intellij-idea-ultimate.desktop',

    'CLICOLOR': '1',

    'CLUTTER_IM_MODULE': 'xim',

    'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus',

    'DESKTOP_SESSION': 'ubuntu',

    'DISPLAY': ':1',

    'GDK_BACKEND': 'x11',

    'GDMSESSION': 'ubuntu',

    'GIO_LAUNCHED_DESKTOP_FILE': '/var/lib/snapd/desktop/applications/intellij-idea-ultimate_intellij-idea-ultimate.desktop',

    'GIO_LAUNCHED_DESKTOP_FILE_PID': '10875',

    'GIT_PAGER': 'cat',

    'GJS_DEBUG_OUTPUT': 'stderr',

    'GJS_DEBUG_TOPICS': 'JS ERROR;JS LOG',

    'GNOME_DESKTOP_SESSION_ID': 'this-is-deprecated',

    'GNOME_SHELL_SESSION_MODE': 'ubuntu',

    'GPG_AGENT_INFO': '/run/user/1000/gnupg/S.gpg-agent:0:1',

    'GTK_IM_MODULE': 'ibus',

    'GTK_MODULES': '',

    'HOME': '/home/jamie',

    'IM_CONFIG_PHASE': '2',

    'JPY_PARENT_PID': '31229',

    'KERNEL_LAUNCH_TIMEOUT': '40',

    'LANG': 'en_GB.UTF-8',

    'LANGUAGE': 'en_GB:en',

    'LOGNAME': 'jamie',

    'MPLBACKEND': 'module://ipykernel.pylab.backend_inline',

    'OLDPWD': '/snap/intellij-idea-ultimate/204/bin',

    'PAGER': 'cat',

    'PATH': './venv/bin:/home/jamie/code/kaggle-bengali-ai/venv/bin:/home/jamie/.local/bin:/usr/local/bin:/usr/local/sbin/:/usr/local/opt/coreutils/libexec/gnubin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:./.anaconda3/bin:./.local/bin:./node_modules/.bin:~/.config/yarn/global/node_modules/.bin/:~/.cabal/bin:~/.cargo/bin:~/.local/bin:~/.rvm/bin:~/Library/Haskell/bin:~/.git-scripts:~/Dropbox/Programming/command-line-scripts:/usr/local/cuda-10.1/bin',

    'PS1': '(venv) ',

    'PWD': '/home/jamie/code/kaggle-bengali-ai',

    'PYTHONPATH': '/home/jamie/code/kaggle-bengali-ai',

    'QT4_IM_MODULE': 'xim',

    'QT_ACCESSIBILITY': '1',

    'QT_IM_MODULE': 'ibus',

    'SESSION_MANAGER': 'local/jamie-Blade-Pro:@/tmp/.ICE-unix/2275,unix/jamie-Blade-Pro:/tmp/.ICE-unix/2275',

    'SHELL': '/bin/bash',

    'SHLVL': '0',

    'SNAP': '/snap/intellij-idea-ultimate/204',

    'SNAP_ARCH': 'amd64',

    'SNAP_COMMON': '/var/snap/intellij-idea-ultimate/common',

    'SNAP_CONTEXT': 'jv7sWxCBpTBHp5RC3XX8yqsmnQgpCRKfcG7QPK6k8Y5V',

    'SNAP_COOKIE': 'jv7sWxCBpTBHp5RC3XX8yqsmnQgpCRKfcG7QPK6k8Y5V',

    'SNAP_DATA': '/var/snap/intellij-idea-ultimate/204',

    'SNAP_INSTANCE_KEY': '',

    'SNAP_INSTANCE_NAME': 'intellij-idea-ultimate',

    'SNAP_LIBRARY_PATH': '/var/lib/snapd/lib/gl:/var/lib/snapd/lib/gl32:/var/lib/snapd/void',

    'SNAP_NAME': 'intellij-idea-ultimate',

    'SNAP_REEXEC': '',

    'SNAP_REVISION': '204',

    'SNAP_USER_COMMON': '/home/jamie/snap/intellij-idea-ultimate/common',

    'SNAP_USER_DATA': '/home/jamie/snap/intellij-idea-ultimate/204',

    'SNAP_VERSION': '2019.3.3',

    'SSH_AGENT_PID': '2386',

    'SSH_AUTH_SOCK': '/run/user/1000/keyring/ssh',

    'TERM': 'xterm-color',

    'TEXTDOMAIN': 'im-config',

    'TEXTDOMAINDIR': '/usr/share/locale/',

    'USER': 'jamie',

    'USERNAME': 'jamie',

    'VIRTUAL_ENV': '/home/jamie/code/kaggle-bengali-ai/venv',

    'WINDOWPATH': '2',

    'XAUTHORITY': '/run/user/1000/gdm/Xauthority',

    'XDG_CONFIG_DIRS': '/etc/xdg/xdg-ubuntu:/etc/xdg',

    'XDG_CURRENT_DESKTOP': 'ubuntu:GNOME',

    'XDG_DATA_DIRS': '/usr/share/ubuntu:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop',

    'XDG_MENU_PREFIX': 'gnome-',

    'XDG_RUNTIME_DIR': '/run/user/1000/snap.intellij-idea-ultimate',

    'XDG_SEAT': 'seat0',

    'XDG_SESSION_DESKTOP': 'ubuntu',

    'XDG_SESSION_ID': '2',

    'XDG_SESSION_TYPE': 'x11',

    'XDG_VTNR': '2',

    'XMODIFIERS': '@im=ibus'

}
# Keys only found on Kaggle and not on Localhost  

(

      ( set(os_environ['kaggle_interactive'].keys())    | set(os_environ['kaggle_commit_batch'].keys()) )

    - ( set(os_environ['localhost_jupyter_lab'].keys()) | set(os_environ['localhost_intellij'].keys())  )

)
# Keys only found on JupyterLab Localhost (run from terminal) and on Kaggle or inside IntelliJ IDE  

(

    set(os_environ['localhost_jupyter_lab'].keys()) 

    - set(os_environ['localhost_intellij'].keys()) 

    - set(os_environ['kaggle_interactive'].keys()) 

    - set(os_environ['kaggle_commit_batch'].keys()) 

)
# Keys only found in localhost IntelliJ  

( 

    set(os_environ['localhost_intellij'].keys())

    - set(os_environ['localhost_jupyter_lab'].keys())   

)
# Keys only found on Localhost (any) and not running on Kaggle  

(

      ( set(os_environ['localhost_jupyter_lab'].keys()) | set(os_environ['localhost_intellij'].keys())  )

    - ( set(os_environ['kaggle_interactive'].keys())    | set(os_environ['kaggle_commit_batch'].keys()) )

)